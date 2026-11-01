5from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import torch

from model import ECGCNN
from samsung_pdf import extract_samsung_ecg


# ================================================================
# Configuration
# ================================================================

PROJECT_ROOT = Path(
    r"D:\ECG-AI-Research-1Lead"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

SAMPLING_RATE = 500

CLASSES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


# ================================================================
# Class-specific thresholds
# ================================================================

THRESHOLDS = {
    "NORM": 0.50,
    "MI": 0.30,
    "STTC": 0.30,
    "CD": 0.35,
    "HYP": 0.25,
}


# ================================================================
# Device
# ================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# Load model
# ================================================================

def load_model():

    model = ECGCNN().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


model, checkpoint = load_model()

BEST_EPOCH = checkpoint["epoch"]

VALIDATION_LOSS = (
    checkpoint["validation_loss"]
)


# ================================================================
# ECG analysis
# ================================================================

def analyze_ecg(pdf_path):

    # ------------------------------------------------------------
    # Extract Samsung ECG waveform
    # ------------------------------------------------------------

    signal = extract_samsung_ecg(
        pdf_path
    )

    # ------------------------------------------------------------
    # Verify signal shape
    # ------------------------------------------------------------

    if signal.shape != torch.Size([5000]):

        raise ValueError(
            "Expected ECG signal with 5000 samples, "
            f"but received {signal.shape}"
        )

    # ------------------------------------------------------------
    # Prepare model input
    #
    # Original:
    #
    # [5000]
    #
    # Add batch dimension:
    #
    # [1, 5000]
    #
    # Add channel dimension:
    #
    # [1, 1, 5000]
    # ------------------------------------------------------------

    model_input = (
        signal
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # ------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------

    with torch.no_grad():

        logits = model(
            model_input
        )

        probabilities = torch.sigmoid(
            logits
        )

    probabilities = (
        probabilities[0]
        .cpu()
    )

    # ------------------------------------------------------------
    # Apply class-specific thresholds
    # ------------------------------------------------------------

    results = []

    for index, class_name in enumerate(
        CLASSES
    ):

        probability = (
            probabilities[index]
            .item()
        )

        threshold = (
            THRESHOLDS[class_name]
        )

        detected = (
            probability >= threshold
        )

        results.append(
            {
                "class": class_name,
                "probability": probability,
                "threshold": threshold,
                "detected": detected,
            }
        )

    return results, signal


# ================================================================
# GUI Application
# ================================================================

class ECGApplication:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "1-Lead ECG AI Analyzer"
        )

        self.root.geometry(
            "900x900"
        )

        self.root.minsize(
            750,
            650
        )

        # --------------------------------------------------------
        # Application state
        # --------------------------------------------------------

        self.selected_pdf = None

        self.current_signal = None

        # --------------------------------------------------------
        # Create GUI
        # --------------------------------------------------------

        self.create_styles()

        self.create_interface()

        self.update_status(
            "Ready — select a Samsung ECG PDF."
        )


    # ============================================================
    # Styles
    # ============================================================

    def create_styles(self):

        style = ttk.Style()

        try:

            style.theme_use(
                "clam"
            )

        except tk.TclError:

            pass

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                24,
                "bold"
            )
        )

        style.configure(
            "Subtitle.TLabel",
            font=(
                "Segoe UI",
                11
            )
        )

        style.configure(
            "Result.TLabel",
            font=(
                "Segoe UI",
                13
            )
        )

        style.configure(
            "Detected.TLabel",
            font=(
                "Segoe UI",
                14,
                "bold"
            )
        )


    # ============================================================
    # Create interface
    # ============================================================

    def create_interface(self):

        # ========================================================
        # OUTER SCROLLABLE AREA
        # ========================================================

        outer_frame = ttk.Frame(
            self.root
        )

        outer_frame.pack(
            fill="both",
            expand=True
        )

        # --------------------------------------------------------
        # Canvas
        # --------------------------------------------------------

        scroll_canvas = tk.Canvas(
            outer_frame,
            highlightthickness=0
        )

        scroll_canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        # --------------------------------------------------------
        # Scrollbar
        # --------------------------------------------------------

        scrollbar = ttk.Scrollbar(
            outer_frame,
            orient="vertical",
            command=scroll_canvas.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )

        scroll_canvas.configure(
            yscrollcommand=scrollbar.set
        )

        # --------------------------------------------------------
        # Main content frame
        # --------------------------------------------------------

        main_frame = ttk.Frame(
            scroll_canvas,
            padding=25
        )

        window_id = (
            scroll_canvas.create_window(
                (0, 0),
                window=main_frame,
                anchor="nw"
            )
        )

        # --------------------------------------------------------
        # Update scrollable region
        # --------------------------------------------------------

        def update_scroll_region(event=None):

            scroll_canvas.configure(
                scrollregion=(
                    scroll_canvas.bbox("all")
                )
            )

        main_frame.bind(
            "<Configure>",
            update_scroll_region
        )

        # --------------------------------------------------------
        # Keep inner frame same width as canvas
        # --------------------------------------------------------

        def resize_inner_frame(event):

            scroll_canvas.itemconfigure(
                window_id,
                width=event.width
            )

        scroll_canvas.bind(
            "<Configure>",
            resize_inner_frame
        )

        # --------------------------------------------------------
        # Mouse wheel
        # --------------------------------------------------------

        def mouse_wheel(event):

            if event.delta != 0:

                scroll_canvas.yview_scroll(
                    int(
                        -1 * (event.delta / 120)
                    ),
                    "units"
                )

        self.root.bind_all(
            "<MouseWheel>",
            mouse_wheel
        )

        # ========================================================
        # HEADER
        # ========================================================

        title = ttk.Label(
            main_frame,
            text="1-Lead ECG AI",
            style="Title.TLabel"
        )

        title.pack(
            pady=(0, 5)
        )

        subtitle = ttk.Label(
            main_frame,
            text="Samsung ECG PDF Analyzer",
            style="Subtitle.TLabel"
        )

        subtitle.pack(
            pady=(0, 20)
        )

        # ========================================================
        # MODEL INFORMATION
        # ========================================================

        model_frame = ttk.LabelFrame(
            main_frame,
            text="Model",
            padding=12
        )

        model_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        model_info = (
            f"Best checkpoint: epoch {BEST_EPOCH}\n"
            f"Validation loss: "
            f"{VALIDATION_LOSS:.6f}\n"
            f"Device: {device}"
        )

        model_label = ttk.Label(
            model_frame,
            text=model_info
        )

        model_label.pack(
            anchor="w"
        )

        # ========================================================
        # PDF SELECTION
        # ========================================================

        file_frame = ttk.LabelFrame(
            main_frame,
            text="Samsung ECG",
            padding=12
        )

        file_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        self.file_label = ttk.Label(
            file_frame,
            text="No ECG PDF selected.",
            wraplength=800
        )

        self.file_label.pack(
            fill="x",
            pady=(0, 10)
        )

        button_frame = ttk.Frame(
            file_frame
        )

        button_frame.pack(
            fill="x"
        )

        self.select_button = ttk.Button(
            button_frame,
            text="Select ECG PDF",
            command=self.select_pdf
        )

        self.select_button.pack(
            side="left"
        )

        self.analyze_button = ttk.Button(
            button_frame,
            text="Analyze ECG",
            command=self.run_analysis,
            state="disabled"
        )

        self.analyze_button.pack(
            side="left",
            padx=10
        )

        # ========================================================
        # STATUS
        # ========================================================

        self.status_label = ttk.Label(
            main_frame,
            text="",
            style="Subtitle.TLabel"
        )

        self.status_label.pack(
            pady=(0, 10)
        )

        # ========================================================
        # ECG WAVEFORM
        # ========================================================

        waveform_frame = ttk.LabelFrame(
            main_frame,
            text="Extracted ECG Waveform — Lead I",
            padding=10
        )

        waveform_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        self.waveform_canvas = tk.Canvas(
            waveform_frame,
            height=260,
            background="white",
            highlightthickness=1,
            highlightbackground="#999999"
        )

        self.waveform_canvas.pack(
            fill="x",
            expand=True
        )

        self.waveform_canvas.bind(
            "<Configure>",
            self.on_waveform_resize
        )

        self.waveform_info = ttk.Label(
            waveform_frame,
            text="No ECG waveform loaded.",
            style="Subtitle.TLabel"
        )

        self.waveform_info.pack(
            anchor="w",
            pady=(7, 0)
        )

        # ========================================================
        # RESULTS
        # ========================================================

        results_frame = ttk.LabelFrame(
            main_frame,
            text="ECG Analysis",
            padding=12
        )

        results_frame.pack(
            fill="x",
            pady=(0, 15)
        )

        # --------------------------------------------------------
        # Result header
        # --------------------------------------------------------

        header = ttk.Frame(
            results_frame
        )

        header.pack(
            fill="x",
            pady=(0, 8)
        )

        ttk.Label(
            header,
            text="Class",
            width=12,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side="left"
        )

        ttk.Label(
            header,
            text="Probability",
            width=15,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side="left"
        )

        ttk.Label(
            header,
            text="Threshold",
            width=15,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side="left"
        )

        ttk.Label(
            header,
            text="Result",
            width=15,
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            side="left"
        )

        # --------------------------------------------------------
        # Result rows
        # --------------------------------------------------------

        self.result_labels = {}

        for class_name in CLASSES:

            row = ttk.Frame(
                results_frame
            )

            row.pack(
                fill="x",
                pady=4
            )

            ttk.Label(
                row,
                text=class_name,
                width=12,
                style="Result.TLabel"
            ).pack(
                side="left"
            )

            probability_label = ttk.Label(
                row,
                text="--",
                width=15,
                style="Result.TLabel"
            )

            probability_label.pack(
                side="left"
            )

            threshold_label = ttk.Label(
                row,
                text=(
                    f"{THRESHOLDS[class_name]:.2f}"
                ),
                width=15,
                style="Result.TLabel"
            )

            threshold_label.pack(
                side="left"
            )

            result_label = ttk.Label(
                row,
                text="--",
                width=15,
                style="Result.TLabel"
            )

            result_label.pack(
                side="left"
            )

            self.result_labels[class_name] = (
                probability_label,
                result_label
            )

        # ========================================================
        # DETECTED PATTERNS
        # ========================================================

        self.detected_label = ttk.Label(
            results_frame,
            text="Detected patterns: --",
            style="Detected.TLabel"
        )

        self.detected_label.pack(
            anchor="w",
            pady=(15, 0)
        )

        # ========================================================
        # FOOTER
        # ========================================================

        footer = ttk.Label(
            main_frame,
            text=(
                "Research prototype • "
                "1-lead ECG classification"
            ),
            style="Subtitle.TLabel"
        )

        footer.pack(
            pady=(10, 0)
        )


    # ============================================================
    # Update status
    # ============================================================

    def update_status(self, text):

        self.status_label.config(
            text=text
        )

        self.root.update_idletasks()


    # ============================================================
    # Select PDF
    # ============================================================

    def select_pdf(self):

        pdf_path = filedialog.askopenfilename(
            title="Select Samsung ECG PDF",
            filetypes=[
                (
                    "PDF files",
                    "*.pdf"
                ),
                (
                    "All files",
                    "*.*"
                ),
            ]
        )

        if not pdf_path:

            return

        self.selected_pdf = Path(
            pdf_path
        )

        self.file_label.config(
            text=str(
                self.selected_pdf
            )
        )

        self.analyze_button.config(
            state="normal"
        )

        self.update_status(
            "ECG selected — ready for analysis."
        )


    # ============================================================
    # Draw ECG waveform
    # ============================================================

    def draw_waveform(self):

        canvas = (
            self.waveform_canvas
        )

        signal = (
            self.current_signal
        )

        if signal is None:

            return

        # --------------------------------------------------------
        # Clear old waveform
        # --------------------------------------------------------

        canvas.delete(
            "all"
        )

        # --------------------------------------------------------
        # Get canvas dimensions
        # --------------------------------------------------------

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width < 100 or height < 100:

            return

        # --------------------------------------------------------
        # Convert tensor to NumPy
        # --------------------------------------------------------

        values = (
            signal.detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------
        # Plot margins
        # --------------------------------------------------------

        left = 55
        right = 15
        top = 15
        bottom = 35

        plot_width = (
            width
            - left
            - right
        )

        plot_height = (
            height
            - top
            - bottom
        )

        # --------------------------------------------------------
        # Signal range
        # --------------------------------------------------------

        signal_min = float(
            values.min()
        )

        signal_max = float(
            values.max()
        )

        signal_range = (
            signal_max
            - signal_min
        )

        if signal_range <= 0:

            signal_range = 1.0

        padding = (
            signal_range * 0.10
        )

        y_min = (
            signal_min
            - padding
        )

        y_max = (
            signal_max
            + padding
        )

        # --------------------------------------------------------
        # Coordinate conversion
        # --------------------------------------------------------

        def x_position(index):

            return (
                left
                + (
                    index
                    / (len(values) - 1)
                )
                * plot_width
            )

        def y_position(value):

            return (
                top
                + (
                    (y_max - value)
                    / (y_max - y_min)
                )
                * plot_height
            )

        # ========================================================
        # Horizontal ECG grid
        # ========================================================

        grid_count = 5

        for i in range(
            grid_count + 1
        ):

            fraction = (
                i / grid_count
            )

            y = (
                top
                + fraction
                * plot_height
            )

            canvas.create_line(
                left,
                y,
                width - right,
                y,
                fill="#e6e6e6"
            )

            value = (
                y_max
                - fraction
                * (y_max - y_min)
            )

            canvas.create_text(
                left - 8,
                y,
                text=f"{value:.2f}",
                anchor="e",
                font=(
                    "Segoe UI",
                    8
                ),
                fill="#555555"
            )

        # ========================================================
        # Vertical time grid
        # ========================================================

        duration = (
            len(values)
            / SAMPLING_RATE
        )

        time_markers = 10

        for i in range(
            time_markers + 1
        ):

            fraction = (
                i / time_markers
            )

            x = (
                left
                + fraction
                * plot_width
            )

            canvas.create_line(
                x,
                top,
                x,
                top + plot_height,
                fill="#eeeeee"
            )

            seconds = (
                fraction
                * duration
            )

            canvas.create_text(
                x,
                height - 15,
                text=f"{seconds:.0f}s",
                anchor="n",
                font=(
                    "Segoe UI",
                    8
                ),
                fill="#555555"
            )

        # ========================================================
        # Zero/reference line
        # ========================================================

        if (
            y_min
            <= 0
            <= y_max
        ):

            zero_y = y_position(
                0.0
            )

            canvas.create_line(
                left,
                zero_y,
                width - right,
                zero_y,
                fill="#999999",
                dash=(4, 4)
            )

        # ========================================================
        # Prepare waveform points
        # ========================================================

        max_display_points = max(
            1000,
            int(plot_width * 3)
        )

        if (
            len(values)
            > max_display_points
        ):

            step = (
                len(values)
                / max_display_points
            )

            indices = [
                min(
                    int(i * step),
                    len(values) - 1
                )
                for i in range(
                    max_display_points
                )
            ]

            plot_values = values[
                indices
            ]

            plot_indices = indices

        else:

            plot_values = values

            plot_indices = range(
                len(values)
            )

        # ========================================================
        # Build Canvas line
        # ========================================================

        points = []

        for index, value in zip(
            plot_indices,
            plot_values
        ):

            points.extend(
                [
                    x_position(index),
                    y_position(
                        float(value)
                    )
                ]
            )

        # ========================================================
        # Draw waveform
        # ========================================================

        if len(points) >= 4:

            canvas.create_line(
                *points,
                fill="#d62828",
                width=1.5,
                smooth=False
            )

        # ========================================================
        # Axis labels
        # ========================================================

        canvas.create_text(
            12,
            top + (
                plot_height / 2
            ),
            text="mV",
            angle=90,
            font=(
                "Segoe UI",
                9,
                "bold"
            )
        )

        canvas.create_text(
            left
            + (
                plot_width / 2
            ),
            height - 2,
            text="Time (seconds)",
            anchor="s",
            font=(
                "Segoe UI",
                9,
                "bold"
            )
        )


    # ============================================================
    # Waveform resize callback
    # ============================================================

    def on_waveform_resize(
        self,
        event=None
    ):

        if self.current_signal is not None:

            self.draw_waveform()


    # ============================================================
    # Run analysis
    # ============================================================

    def run_analysis(self):

        if self.selected_pdf is None:

            messagebox.showwarning(
                "No ECG selected",
                "Please select a Samsung ECG PDF first."
            )

            return

        # --------------------------------------------------------
        # Disable buttons during analysis
        # --------------------------------------------------------

        self.analyze_button.config(
            state="disabled"
        )

        self.select_button.config(
            state="disabled"
        )

        self.update_status(
            "Extracting ECG waveform..."
        )

        try:

            # ----------------------------------------------------
            # Run extraction + AI prediction
            # ----------------------------------------------------

            results, signal = analyze_ecg(
                self.selected_pdf
            )

            # ----------------------------------------------------
            # Store signal
            # ----------------------------------------------------

            self.current_signal = signal

            # ----------------------------------------------------
            # Draw waveform
            # ----------------------------------------------------

            self.draw_waveform()

            # ----------------------------------------------------
            # ECG statistics
            # ----------------------------------------------------

            minimum = (
                signal.min().item()
            )

            maximum = (
                signal.max().item()
            )

            mean = (
                signal.mean().item()
            )

            standard_deviation = (
                signal.std().item()
            )

            self.waveform_info.config(
                text=(
                    f"5000 samples • "
                    f"{SAMPLING_RATE} Hz • "
                    f"10 seconds • "
                    f"Min: {minimum:.3f} mV • "
                    f"Max: {maximum:.3f} mV • "
                    f"Mean: {mean:.3f} mV • "
                    f"Std: {standard_deviation:.3f} mV"
                )
            )

            self.update_status(
                "ECG waveform extracted — "
                "AI analysis complete."
            )

            # ====================================================
            # Update prediction results
            # ====================================================

            detected_classes = []

            for result in results:

                class_name = (
                    result["class"]
                )

                probability = (
                    result["probability"]
                )

                detected = (
                    result["detected"]
                )

                (
                    probability_label,
                    result_label
                ) = self.result_labels[
                    class_name
                ]

                probability_label.config(
                    text=(
                        f"{probability * 100:.2f}%"
                    )
                )

                if detected:

                    result_label.config(
                        text="✓ DETECTED"
                    )

                    detected_classes.append(
                        class_name
                    )

                else:

                    result_label.config(
                        text="—"
                    )

            # ====================================================
            # Detected pattern summary
            # ====================================================

            if detected_classes:

                patterns = ", ".join(
                    detected_classes
                )

                self.detected_label.config(
                    text=(
                        f"Detected patterns: "
                        f"{patterns}"
                    )
                )

            else:

                self.detected_label.config(
                    text=(
                        "Detected patterns: None"
                    )
                )

        except Exception as error:

            self.update_status(
                "Analysis failed."
            )

            messagebox.showerror(
                "ECG Analysis Error",
                str(error)
            )

        finally:

            self.analyze_button.config(
                state="normal"
            )

            self.select_button.config(
                state="normal"
            )


# ================================================================
# Main
# ================================================================

def main():

    print(
        "Starting 1-Lead ECG AI GUI..."
    )

    print(
        "Using device:",
        device
    )

    print(
        "Best model epoch:",
        BEST_EPOCH
    )

    print(
        "Validation loss:",
        f"{VALIDATION_LOSS:.6f}"
    )

    root = tk.Tk()

    application = ECGApplication(
        root
    )

    root.mainloop()


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    main()
    