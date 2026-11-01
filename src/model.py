import torch
import torch.nn as nn


class ECGCNN(nn.Module):
    """
    1-lead ECG convolutional neural network.

    Input:
        (batch, 1, 5000)

    Output:
        (batch, 5)

    Classes:
        NORM
        MI
        STTC
        CD
        HYP
    """

    def __init__(self):

        super().__init__()

        # ========================================================
        # Convolutional feature extractor
        # ========================================================

        self.features = nn.Sequential(

            # ----------------------------------------------------
            # Block 1
            # ----------------------------------------------------

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ----------------------------------------------------
            # Block 2
            # ----------------------------------------------------

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ----------------------------------------------------
            # Block 3
            # ----------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ----------------------------------------------------
            # Block 4
            # ----------------------------------------------------

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),
        )

        # ========================================================
        # Global average pooling
        # ========================================================

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ========================================================
        # Classifier
        # ========================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                5
            )
        )

    # ============================================================
    # Forward pass
    # ============================================================

    def forward(self, x):

        x = self.features(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ================================================================
# Model test
# ================================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------

    model = ECGCNN().to(device)

    print(
        "Model device:",
        next(model.parameters()).device
    )

    # ------------------------------------------------------------
    # Create fake 1-lead ECG batch
    # ------------------------------------------------------------

    test_input = torch.randn(
        32,
        1,
        5000
    ).to(device)

    print(
        "Input shape:",
        test_input.shape
    )

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    output = model(test_input)

    print(
        "Output shape:",
        output.shape
    )

    print(
        "Output device:",
        output.device
    )

    print("Model test successful.")