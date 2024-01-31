# functions copied from https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
#
# TODO this is hacky solution and the efficientnet_lite need to be refactored for use with newer TF
# and maybe our models need to refactored to Keras that has been moved again to the new repository
import warnings
from tensorflow.keras import backend
from tensorflow.keras import activations

def obtain_input_shape(
    input_shape,
    default_size,
    min_size,
    data_format,
    require_flatten,
    weights=None,
):
    """Internal utility to compute/validate a model's input shape.

    Args:
      input_shape: Either None (will return the default network input shape),
        or a user-provided shape to be validated.
      default_size: Default input width/height for the model.
      min_size: Minimum input width/height accepted by the model.
      data_format: Image data format to use.
      require_flatten: Whether the model is expected to
        be linked to a classifier via a Flatten layer.
      weights: One of `None` (random initialization)
        or 'imagenet' (pre-training on ImageNet).
        If weights='imagenet' input channels must be equal to 3.

    Returns:
      An integer shape tuple (may include None entries).

    Raises:
      ValueError: In case of invalid argument values.
    """
    if weights != "imagenet" and input_shape and len(input_shape) == 3:
        if data_format == "channels_first":
            correct_channel_axis = 1 if len(input_shape) == 4 else 0
            if input_shape[correct_channel_axis] not in {1, 3}:
                warnings.warn(
                    "This model usually expects 1 or 3 input channels. "
                    "However, it was passed an input_shape "
                    f"with {input_shape[0]} input channels.",
                    stacklevel=2,
                )
            default_shape = (input_shape[0], default_size, default_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    "This model usually expects 1 or 3 input channels. "
                    "However, it was passed an input_shape "
                    f"with {input_shape[-1]} input channels.",
                    stacklevel=2,
                )
            default_shape = (default_size, default_size, input_shape[-1])
    else:
        if data_format == "channels_first":
            default_shape = (3, default_size, default_size)
        else:
            default_shape = (default_size, default_size, 3)
    if weights == "imagenet" and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError(
                    "When setting `include_top=True` "
                    "and loading `imagenet` weights, "
                    f"`input_shape` should be {default_shape}.  "
                    f"Received: input_shape={input_shape}"
                )
        return default_shape
    if input_shape:
        if data_format == "channels_first":
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        "`input_shape` must be a tuple of three integers."
                    )
                if input_shape[0] != 3 and weights == "imagenet":
                    raise ValueError(
                        "The input must have 3 channels; Received "
                        f"`input_shape={input_shape}`"
                    )
                if (
                    input_shape[1] is not None and input_shape[1] < min_size
                ) or (input_shape[2] is not None and input_shape[2] < min_size):
                    raise ValueError(
                        f"Input size must be at least {min_size}"
                        f"x{min_size}; Received: "
                        f"input_shape={input_shape}"
                    )
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError(
                        "`input_shape` must be a tuple of three integers."
                    )
                if input_shape[-1] != 3 and weights == "imagenet":
                    raise ValueError(
                        "The input must have 3 channels; Received "
                        f"`input_shape={input_shape}`"
                    )
                if (
                    input_shape[0] is not None and input_shape[0] < min_size
                ) or (input_shape[1] is not None and input_shape[1] < min_size):
                    raise ValueError(
                        "Input size must be at least "
                        f"{min_size}x{min_size}; Received: "
                        f"input_shape={input_shape}"
                    )
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == "channels_first":
                input_shape = (3, None, None)
            else:
                input_shape = (None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError(
                "If `include_top` is True, "
                "you should specify a static `input_shape`. "
                f"Received: input_shape={input_shape}"
            )
    return input_shape


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """
    img_dim = 2 if backend.image_data_format() == "channels_first" else 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def validate_activation(classifier_activation, weights):
    """validates that the classifer_activation is compatible with the weights.

    Args:
      classifier_activation: str or callable activation function
      weights: The pretrained weights to load.

    Raises:
      ValueError: if an activation other than `None` or `softmax` are used with
        pretrained weights.
    """
    if weights is None:
        return

    classifier_activation = activations.get(classifier_activation)
    if classifier_activation not in {
        activations.get("softmax"),
        activations.get(None),
    }:
        raise ValueError(
            "Only `None` and `softmax` activations are allowed "
            "for the `classifier_activation` argument when using "
            "pretrained weights, with `include_top=True`; Received: "
            f"classifier_activation={classifier_activation}"
        )
    