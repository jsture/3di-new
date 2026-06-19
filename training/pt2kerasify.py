"""Converts a trained PyTorch encoder model to Keras format and exports it using Kerasify."""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn

try:
    import keras
    import kerasify
except ImportError:
    # Handle environment import failures gracefully during structural modernization
    keras = None
    kerasify = None


def convert_pt_to_keras(encoder: nn.Sequential) -> "keras.models.Sequential":
    """Convert a PyTorch nn.Sequential encoder to a Keras Sequential model.

    Args:
        encoder: PyTorch nn.Sequential encoder model containing only Linear and ReLU layers.

    Returns:
        Keras Sequential model with copied weights.
    """
    if keras is None:
        raise ImportError(
            "Keras is not installed. Please install keras to convert models."
        )

    # Determine shapes and node dimensions
    input_shape = [encoder[0].in_features]
    n_units_lst = [
        layer.out_features
        for layer in encoder
        if not isinstance(layer, nn.ReLU)
    ]

    # Re-construct identical architecture in Keras
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            n_units_lst[0], input_shape=input_shape, activation="relu"
        )
    )
    for n_units in n_units_lst[1:-1]:
        model.add(keras.layers.Dense(n_units, activation="relu"))
    model.add(keras.layers.Dense(n_units_lst[-1], activation="linear"))

    print("\nTarget Keras Model Summary:")
    model.summary()

    # Transpose and load PyTorch weights into corresponding Keras layers
    n_layers = len(n_units_lst)
    for i in range(n_layers):
        w_tensor = encoder[i * 2].weight.detach().cpu().numpy().T
        b_tensor = encoder[i * 2].bias.detach().cpu().numpy()
        model.layers[i].set_weights([w_tensor, b_tensor])

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch sequential model to Kerasify format."
    )
    parser.add_argument("pt_model", type=str, help="Path to the source PyTorch .pt model file.")
    parser.add_argument("out_model", type=str, help="Path to write the exported Kerasify model file.")
    args = parser.parse_args()

    if keras is None or kerasify is None:
        print(
            "Error: Keras or Kerasify packages are missing. Model conversion aborted.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Load PyTorch model on CPU
        encoder = torch.load(args.pt_model, map_location="cpu")
        if not isinstance(encoder, nn.Sequential):
            raise TypeError("Loaded PyTorch model must be of type nn.Sequential")

        # Convert to Keras
        keras_model = convert_pt_to_keras(encoder)

        # Save using Kerasify
        kerasify.export_model(keras_model, args.out_model)
        print(f"Model successfully exported to Kerasify format at: {args.out_model}")

    except Exception as e:
        print(f"Error converting model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
