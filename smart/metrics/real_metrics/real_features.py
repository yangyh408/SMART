import math
from .geometry_utils import *
import tensorflow as tf
import numpy as np
from .custom_metrics import MetricFeatures

# Constant distance to apply when distances between objects are invalid. This
# will avoid the propagation of nans and should be reduced out when taking the
# minimum anyway.
EXTREMELY_LARGE_DISTANCE = 1e10
# Collision threshold, i.e. largest distance between objects that is considered
# to be a collision.
COLLISION_DISTANCE_THRESHOLD = 0.0
# Rounding factor to apply to the corners of the object boxes in distance and
# collision computation. The rounding factor is between 0 and 1, where 0 yields
# rectangles with sharp corners (no rounding) and 1 yields capsule shapes.
# Default value of 0.7 conservately fits most vehicle contours.
CORNER_ROUNDING_FACTOR = 0.7

# Condition thresholds for filtering obstacles driving ahead of the ego pbject
# when computing the time-to-collision metric. This metric only considers
# collisions in lane-following a.k.a. tailgating situations.
# Maximum allowed difference in heading.
MAX_HEADING_DIFF = math.radians(75.0)  # radians.
MAX_LONG_DIFF = 40 # meters
# Maximum allowed difference in heading in case of small lateral overlap.
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)  # radians.
# Lateral overlap threshold below which the tighter heading alignment condition
# `_MAX_HEADING_DIFF_FOR_SMALL_OVERLAP` is used.
SMALL_OVERLAP_THRESHOLD = 0.5  # meters.

# Maximum time-to-collision, in seconds, used to clip large values or in place
# of invalid values.
MAXIMUM_TIME_TO_COLLISION = 5.0

def _central_diff(t: tf.Tensor, pad_value: float) -> tf.Tensor:
    # Prepare the tensor containing the value(s) to pad the result with.
    pad_shape = (*t.shape[:-1], 1)
    pad_tensor = tf.fill(pad_shape, pad_value)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return tf.concat([pad_tensor, diff_t, pad_tensor], axis=-1)

def _wrap_angle(angle: tf.Tensor) -> tf.Tensor:
    """Wraps angles in the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_kinematic_features(
    x: tf.Tensor,
    y: tf.Tensor,
    heading: tf.Tensor,
    valid: tf.Tensor,
    seconds_per_step: float
):
    # 将不可用部分坐标和航向值设置为nan
    x = tf.where(valid, x, np.nan)
    y = tf.where(valid, y, np.nan)
    heading = tf.where(valid, heading, np.nan)
    
    # Linear speed and acceleration.
    dpos = _central_diff(tf.stack([x, y], axis=0), pad_value=np.nan)
    linear_speed = tf.linalg.norm(
        dpos, ord='euclidean', axis=0) / seconds_per_step
    linear_accel = _central_diff(linear_speed, pad_value=np.nan) / seconds_per_step
    dh_step = _wrap_angle(_central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = _wrap_angle(_central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step**2)
    return linear_speed, linear_accel, dh, d2h

def _get_object_following_mask(
    longitudinal_distance: tf.Tensor,
    lateral_overlap: tf.Tensor,
    yaw_diff: tf.Tensor,
) -> tf.Tensor:
    # Check object is ahead of the ego box's front.
    valid_mask = longitudinal_distance > 0.0
    valid_mask = tf.logical_and(valid_mask, longitudinal_distance <= MAX_LONG_DIFF)

    # Check alignment.
    valid_mask = tf.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)

    # Check object is directly ahead of the ego box.
    valid_mask = tf.logical_and(valid_mask, lateral_overlap < 0.0)

    # Check strict alignment if the overlap is small.
    # `lateral_overlap` is a signed penetration distance: it is negative when the
    # boxes have an actual lateral overlap.
    return tf.logical_and(
        valid_mask,
        tf.logical_or(
            lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
            yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
        ),
    )

def find_ahead_index(
        boxes: tf.Tensor,
        valid: tf.Tensor
):
    eval_boxes = boxes
    ego_xy, ego_sizes, ego_yaw, ego_speed = tf.split(
        eval_boxes, num_or_size_splits=[2, 2, 1, 1], axis=-1
    )
    other_xy, other_sizes, other_yaw, _ = tf.split(
        boxes, num_or_size_splits=[2, 2, 1, 1], axis=-1
    )

    yaw_diff = tf.math.abs(other_yaw[:, tf.newaxis] - ego_yaw[:, :, tf.newaxis])
    yaw_diff_cos = tf.math.cos(yaw_diff)
    yaw_diff_sin = tf.math.sin(yaw_diff)

    # 其余车辆在被查询车辆坐标系下的车身最大纵向长度投影
    other_long_offset = dot_product_2d(
        other_sizes[:, tf.newaxis] / 2.0,
        tf.math.abs(tf.concat([yaw_diff_cos, yaw_diff_sin], axis=-1)),
    )
    # 其余车辆在被查询车辆坐标系下的车身最大横向宽度投影
    other_lat_offset = dot_product_2d(
        other_sizes[:, tf.newaxis] / 2.0,
        tf.math.abs(tf.concat([yaw_diff_sin, yaw_diff_cos], axis=-1)),
    )

    # 将其余车辆的坐标转换到被查询车辆坐标系下
    other_relative_xy = rotate_2d_points(
        (other_xy[:, tf.newaxis] - ego_xy[:, :, tf.newaxis]),
        -ego_yaw,
    )

    # 纵向距离差异
    long_distance = (
        other_relative_xy[..., 0]
        - ego_sizes[:, :, tf.newaxis, 0] / 2.0
        - other_long_offset
    )

    # 横向车身覆盖度，lat_overlap小于0说明该车与被查询车辆在横向有重叠
    lat_overlap = (
        tf.math.abs(other_relative_xy[..., 1])
        - ego_sizes[:, :, tf.newaxis, 1] / 2.0
        - other_lat_offset
    )
    
    following_mask = _get_object_following_mask(
        long_distance,
        lat_overlap,
        yaw_diff[..., 0],
    )

    # Mask out boxes that are invalid or don't satisfy "following" conditions.
    valid_mask = tf.logical_and(
        tf.repeat(valid[..., tf.newaxis], valid.shape[-1], axis=-1), 
        following_mask
    )

    # `masked_long_distance` shape: (num_steps, num_eval_objects, num_objects)
    # masked_long_distance = (
    #     long_distance
    #     + (1.0 - tf.cast(valid_mask, tf.float32)) * EXTREMELY_LARGE_DISTANCE
    # )
    masked_long_distance = tf.where(
        valid_mask,
        long_distance,
        EXTREMELY_LARGE_DISTANCE
    )

    # `box_ahead_index` shape: (num_steps, num_evaluated_objects)
    box_ahead_index = tf.math.argmin(masked_long_distance, axis=-1)

    return masked_long_distance, box_ahead_index


def compute_real_metric_features(
        center_x: tf.Tensor,
        center_y: tf.Tensor,
        length: tf.Tensor,
        width: tf.Tensor,
        heading: tf.Tensor,
        valid: tf.Tensor,
):
    linear_speed, linear_accel, yaw_speed, yaw_accel = compute_kinematic_features(center_x, center_y, heading, valid, 0.1)

    boxes = tf.stack([center_x, center_y, length, width, heading, linear_speed], axis=-1)
    masked_long_distance, box_ahead_index = find_ahead_index(
        tf.transpose(boxes, perm=[1, 0, 2]),
        tf.transpose(valid, perm=[1, 0])
    )
    
    mask_without_ahead_index = tf.reduce_all(masked_long_distance >= EXTREMELY_LARGE_DISTANCE, axis=-1)
    nan_tensor = tf.fill(tf.shape(box_ahead_index), tf.constant(float('nan'), dtype=tf.float32)) 

    # 有效的前车编号
    # valid_ahead_index = tf.where(mask_without_ahead_index, -1, box_ahead_index)
    # for i, j in enumerate(valid_ahead_index[0]):
    #     if j != -1:
    #         print(f"{i} -> {j}")
            
    # 与前车相对距离提取
    distance_to_box_ahead = tf.gather(
        masked_long_distance, box_ahead_index, batch_dims=2
    )
    rel_distance = tf.where(mask_without_ahead_index, nan_tensor, distance_to_box_ahead)

    # 与前车相对速度提取
    speed_to_box_ahead = tf.gather(
        tf.broadcast_to(
            tf.transpose(
                linear_speed[
                    :,
                    tf.newaxis,
                    :,
                ]
            ),
            masked_long_distance.shape,
        ),
        box_ahead_index,
        batch_dims=2,
    )
    speed_to_box_ahead = tf.where(mask_without_ahead_index, nan_tensor, speed_to_box_ahead)
    rel_speed = tf.transpose(linear_speed) - speed_to_box_ahead

    # 与前车相对加速度提取
    acc_to_box_ahead = tf.gather(
        tf.broadcast_to(
            tf.transpose(
                linear_accel[
                    :,
                    tf.newaxis,
                    :,
                ]
            ),
            masked_long_distance.shape,
        ),
        box_ahead_index,
        batch_dims=2,
    )
    acc_to_box_ahead = tf.where(mask_without_ahead_index, nan_tensor, acc_to_box_ahead)
    rel_acc = tf.transpose(linear_accel) - acc_to_box_ahead

    # TTC计算
    time_to_collision = tf.where(
        tf.logical_and(~np.isnan(rel_speed), rel_speed < 0.0),
        MAXIMUM_TIME_TO_COLLISION,
        rel_distance / rel_speed,
    )
    
    return MetricFeatures(
        linear_accel = linear_accel,
        linear_speed = linear_speed,
        yaw_speed = yaw_speed,
        yaw_accel = yaw_accel,
        relative_distance = tf.transpose(rel_distance),
        relative_speed = tf.transpose(rel_speed),
        relative_accel = tf.transpose(rel_acc),
        ttc = tf.transpose(time_to_collision)
    )

