from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
try:
    from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressDialog,
        QPushButton,
        QSplitter,
        QSpinBox,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    try:
        from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
        from PyQt6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QProgressDialog,
            QPushButton,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        from PySide6.QtCore import QObject, Qt, QThread, Signal as pyqtSignal
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QProgressDialog,
            QPushButton,
            QSplitter,
            QSpinBox,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

try:
    QT_CHECKED = int(Qt.Checked)
except AttributeError:
    QT_CHECKED = int(Qt.CheckState.Checked)
from scipy.io import loadmat

try:
    from .core import (
        build_legend,
        constellation_symbols,
        getCapacity,
        getParameters,
        ldpc_matrices_available,
        load_image_bits,
        simulate_image,
        simulate_system,
    )
except ImportError:
    from core import (
        build_legend,
        constellation_symbols,
        getCapacity,
        getParameters,
        ldpc_matrices_available,
        load_image_bits,
        simulate_image,
        simulate_system,
    )


class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 5.5, height: float = 3.6, dpi: int = 100) -> None:
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class SimulationWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object, bool)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def _emit_progress(self, done: int, total: int, msg: str) -> None:
        self.progress.emit(int(done), int(total), msg)

    def run(self) -> None:
        try:
            if self.cfg["imagen"]:
                results = simulate_image(
                    sourceBits=self.cfg["sourceBits"],
                    image_shape=self.cfg["image_shape"],
                    modulations=self.cfg["modulations"],
                    order=self.cfg["order"],
                    coding=self.cfg["coding"],
                    infoCoding=self.cfg["infoCoding"],
                    channelType=self.cfg["channelType"],
                    numAntennas=self.cfg["numAntennas"],
                    SNRdB=self.cfg["SNRdB"],
                    progress_callback=self._emit_progress,
                    cancel_check=self._is_cancelled,
                )
                if self._cancelled:
                    self.cancelled.emit()
                else:
                    self.finished.emit(results, True)
            else:
                results = simulate_system(
                    sourceBits=self.cfg["sourceBits"],
                    modulations=self.cfg["modulations"],
                    order=self.cfg["order"],
                    coding=self.cfg["coding"],
                    infoCoding=self.cfg["infoCoding"],
                    channelType=self.cfg["channelType"],
                    numAntennas=self.cfg["numAntennas"],
                    SNRdB=self.cfg["SNRdB"],
                    showConst=self.cfg["showConst"],
                    progress_callback=self._emit_progress,
                    cancel_check=self._is_cancelled,
                )
                if self._cancelled:
                    self.cancelled.emit()
                else:
                    self.finished.emit(results, False)
        except InterruptedError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HESCOD - Python + PyQt")
        self.resize(1400, 900)

        # Sensible defaults so simulation can run immediately.
        self.modulations = [False, False, False, True, False, False, False, False, False]  # QPSK
        self.order = "gray"
        self.numBits = 1_000_000
        self.sourceBits = np.array([], dtype=np.uint8)
        self.image_shape: Optional[Tuple[int, int]] = None
        self.minSNR = 0
        self.maxSNR = 12
        self.codingMethod = [True, False, False, False, False]
        self.infoCoding: List[Any] = [
            [True, False, False],
            [7, 5],
            [True, False, False, False],
            [True, False, False],
            3,
        ]
        self.imagen = False
        self.show_const_received = False
        self.channelType = [True, False, False, False]
        self.numAntenas = [False, False, False]
        self._sim_thread: Optional[QThread] = None
        self._sim_worker: Optional[SimulationWorker] = None
        self._progress_dialog: Optional[QProgressDialog] = None

        self._build_ui()
        self._show_logo()
        self._apply_defaults()

    def _apply_defaults(self) -> None:
        self.edit_nbits.setText("1e6")
        self.sp_snr_min.setValue(self.minSNR)
        self.sp_snr_max.setValue(self.maxSNR)

        self.edit_conv.blockSignals(True)
        self.edit_conv.setText(" ".join(str(x) for x in self.infoCoding[1]))
        self.edit_conv.blockSignals(False)

        self.edit_rs_k.blockSignals(True)
        self.edit_rs_k.setText(str(self.infoCoding[4]))
        self.edit_rs_k.blockSignals(False)

        for i, cb in enumerate(self.cb_mod):
            cb.setChecked(bool(self.modulations[i]))

        for cb, val in [
            (self.cb_no, self.codingMethod[0]),
            (self.cb_hamming, self.codingMethod[1]),
            (self.cb_conv, self.codingMethod[2]),
            (self.cb_ldpc, self.codingMethod[3]),
            (self.cb_rs, self.codingMethod[4]),
            (self.cb_ham_l2, self.infoCoding[0][0]),
            (self.cb_ham_l3, self.infoCoding[0][1]),
            (self.cb_ham_l4, self.infoCoding[0][2]),
            (self.cb_ld_12, self.infoCoding[2][0]),
            (self.cb_ld_23, self.infoCoding[2][1]),
            (self.cb_ld_34, self.infoCoding[2][2]),
            (self.cb_ld_56, self.infoCoding[2][3]),
            (self.cb_rs_l3, self.infoCoding[3][0]),
            (self.cb_rs_l4, self.infoCoding[3][1]),
            (self.cb_rs_l5, self.infoCoding[3][2]),
            (self.cb_awgn, self.channelType[0]),
            (self.cb_rayleigh, self.channelType[1]),
            (self.cb_mimo, self.channelType[2]),
            (self.cb_veh, self.channelType[3]),
            (self.cb_n2, self.numAntenas[0]),
            (self.cb_n4, self.numAntenas[1]),
            (self.cb_n8, self.numAntenas[2]),
        ]:
            cb.setChecked(bool(val))

        self._update_coding_option_widgets()
        self._update_mimo_option_widgets()

        self.on_generate_bits()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        top_buttons = QHBoxLayout()
        main.addLayout(top_buttons)

        self.btn_sim = QPushButton("Simular Transmision")
        self.btn_adapt = QPushButton("Codificacion Adaptativa")
        self.btn_sim.clicked.connect(self.on_simulate)
        self.btn_adapt.clicked.connect(self.on_plot_adapt)
        top_buttons.addWidget(self.btn_sim)
        top_buttons.addWidget(self.btn_adapt)
        top_buttons.addStretch(1)

        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter, 1)

        left_panel = QWidget()
        left_col = QVBoxLayout(left_panel)
        self.canvas = MplCanvas(width=6.0, height=4.2)
        left_col.addWidget(self.canvas, 1)
        left_col.addWidget(self._build_info_group())
        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_col = QVBoxLayout(right_panel)
        right_col.addWidget(self._build_source_group())
        right_col.addWidget(self._build_mod_group())
        right_col.addWidget(self._build_coding_group())
        right_col.addWidget(self._build_channel_group())
        right_col.addStretch(1)
        splitter.addWidget(right_panel)

        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _build_source_group(self) -> QGroupBox:
        box = QGroupBox("Generar Bits Fuente")
        lay = QGridLayout(box)

        self.edit_nbits = QLineEdit("1e6")
        self.edit_nbits.setPlaceholderText("Ej: 1000000 o 1e6")

        self.btn_generate = QPushButton("Generar")
        self.btn_load = QPushButton("Cargar Archivo")
        self.btn_generate.clicked.connect(self.on_generate_bits)
        self.btn_load.clicked.connect(self.on_load_image)
        self.lbl_bits_status = QLabel("")

        lay.addWidget(QLabel("Nº Bits"), 0, 0)
        lay.addWidget(self.edit_nbits, 0, 1)
        lay.addWidget(self.btn_generate, 0, 2)
        lay.addWidget(self.btn_load, 1, 0, 1, 3)
        lay.addWidget(self.lbl_bits_status, 2, 0, 1, 3)
        return box

    def _build_mod_group(self) -> QGroupBox:
        box = QGroupBox("Modulaciones")
        lay = QGridLayout(box)

        labels = [
            "2 ASK",
            "4 ASK",
            "8 ASK",
            "QPSK",
            "8 PSK",
            "16 PSK",
            "16 QAM",
            "64 QAM",
            "256 QAM",
        ]
        self.cb_mod: List[QCheckBox] = []

        for i, lab in enumerate(labels):
            r = i % 3
            c = (i // 3) * 2
            cb = QCheckBox(lab)
            cb.stateChanged.connect(lambda state, idx=i: self.on_mod_changed(idx, state))
            cb.setChecked(bool(self.modulations[i]))
            self.cb_mod.append(cb)
            lay.addWidget(cb, r, c + 1)

            b = QPushButton("i")
            b.setFixedSize(28, 28)
            b.clicked.connect(lambda _, idx=i: self.on_show_constellation(idx))
            lay.addWidget(b, r, c)

        self.combo_order = QComboBox()
        self.combo_order.addItems(["Gray Mapping", "Natural"])
        self.combo_order.currentTextChanged.connect(self.on_order_changed)
        lay.addWidget(QLabel("Ordenamiento"), 3, 0, 1, 2)
        lay.addWidget(self.combo_order, 3, 2, 1, 4)

        return box

    def _build_coding_group(self) -> QGroupBox:
        box = QGroupBox("Codificacion")
        lay = QVBoxLayout(box)

        tabs = QTabWidget()
        lay.addWidget(tabs)

        # No coding tab
        tab_no = QWidget()
        no_lay = QVBoxLayout(tab_no)
        self.cb_no = QCheckBox("Sin Codificacion")
        self.cb_no.setChecked(True)
        self.cb_no.stateChanged.connect(lambda s: self.on_code_changed(0, s))
        self.cb_show_const = QCheckBox("Mostrar Constelacion Simbolos Recibidos")
        self.cb_show_const.stateChanged.connect(self.on_show_const_changed)
        no_lay.addWidget(self.cb_no)
        no_lay.addWidget(self.cb_show_const)
        no_lay.addStretch(1)
        tabs.addTab(tab_no, "No Cod.")

        # Hamming
        tab_h = QWidget()
        h_lay = QHBoxLayout(tab_h)
        self.cb_hamming = QCheckBox("Hamming")
        self.cb_hamming.stateChanged.connect(lambda s: self.on_code_changed(1, s))
        h_lay.addWidget(self.cb_hamming)
        self.cb_ham_l2 = QCheckBox("L = 2")
        self.cb_ham_l3 = QCheckBox("L = 3")
        self.cb_ham_l4 = QCheckBox("L = 4")
        self.cb_ham_l2.setChecked(True)
        self.cb_ham_l2.stateChanged.connect(lambda s: self.on_hamming_l_changed(0, s))
        self.cb_ham_l3.stateChanged.connect(lambda s: self.on_hamming_l_changed(1, s))
        self.cb_ham_l4.stateChanged.connect(lambda s: self.on_hamming_l_changed(2, s))
        h_lay.addWidget(self.cb_ham_l2)
        h_lay.addWidget(self.cb_ham_l3)
        h_lay.addWidget(self.cb_ham_l4)
        h_lay.addStretch(1)
        tabs.addTab(tab_h, "Hamming")

        # RS
        tab_rs = QWidget()
        rs_lay = QHBoxLayout(tab_rs)
        self.cb_rs = QCheckBox("Reed-Solomon")
        self.cb_rs.stateChanged.connect(lambda s: self.on_code_changed(4, s))
        rs_lay.addWidget(self.cb_rs)

        self.cb_rs_l3 = QCheckBox("L = 3")
        self.cb_rs_l4 = QCheckBox("L = 4")
        self.cb_rs_l5 = QCheckBox("L = 5")
        self.cb_rs_l3.setChecked(True)
        self.cb_rs_l3.stateChanged.connect(lambda s: self.on_rs_l_changed(0, s))
        self.cb_rs_l4.stateChanged.connect(lambda s: self.on_rs_l_changed(1, s))
        self.cb_rs_l5.stateChanged.connect(lambda s: self.on_rs_l_changed(2, s))
        rs_lay.addWidget(self.cb_rs_l3)
        rs_lay.addWidget(self.cb_rs_l4)
        rs_lay.addWidget(self.cb_rs_l5)

        rs_lay.addWidget(QLabel("k"))
        self.edit_rs_k = QLineEdit("3")
        self.edit_rs_k.setFixedWidth(50)
        self.edit_rs_k.textChanged.connect(self.on_rs_k_changed)
        rs_lay.addWidget(self.edit_rs_k)
        rs_lay.addStretch(1)
        tabs.addTab(tab_rs, "RS")

        # Convolutional
        tab_conv = QWidget()
        cv_lay = QHBoxLayout(tab_conv)
        self.cb_conv = QCheckBox("Convolucional")
        self.cb_conv.stateChanged.connect(lambda s: self.on_code_changed(2, s))
        cv_lay.addWidget(self.cb_conv)
        cv_lay.addWidget(QLabel("Matriz Generadora"))
        self.edit_conv = QLineEdit("1 1")
        self.edit_conv.textChanged.connect(self.on_conv_changed)
        cv_lay.addWidget(self.edit_conv)
        cv_lay.addStretch(1)
        tabs.addTab(tab_conv, "Convolucional")

        # LDPC
        tab_ldpc = QWidget()
        ld_lay = QHBoxLayout(tab_ldpc)
        self.cb_ldpc = QCheckBox("LDPC")
        self.cb_ldpc.stateChanged.connect(lambda s: self.on_code_changed(3, s))
        ld_lay.addWidget(self.cb_ldpc)
        self.cb_ld_12 = QCheckBox("1/2")
        self.cb_ld_23 = QCheckBox("2/3")
        self.cb_ld_34 = QCheckBox("3/4")
        self.cb_ld_56 = QCheckBox("5/6")
        self.cb_ld_12.setChecked(True)
        self.cb_ld_12.stateChanged.connect(lambda s: self.on_ldpc_rate_changed(0, s))
        self.cb_ld_23.stateChanged.connect(lambda s: self.on_ldpc_rate_changed(1, s))
        self.cb_ld_34.stateChanged.connect(lambda s: self.on_ldpc_rate_changed(2, s))
        self.cb_ld_56.stateChanged.connect(lambda s: self.on_ldpc_rate_changed(3, s))
        ld_lay.addWidget(self.cb_ld_12)
        ld_lay.addWidget(self.cb_ld_23)
        ld_lay.addWidget(self.cb_ld_34)
        ld_lay.addWidget(self.cb_ld_56)
        ld_lay.addStretch(1)
        tabs.addTab(tab_ldpc, "LDPC")

        return box

    def _build_channel_group(self) -> QGroupBox:
        box = QGroupBox("Canal")
        lay = QGridLayout(box)

        self.sp_snr_min = QSpinBox()
        self.sp_snr_max = QSpinBox()
        self.sp_snr_min.setRange(-50, 80)
        self.sp_snr_max.setRange(-50, 80)
        self.sp_snr_min.setValue(self.minSNR)
        self.sp_snr_max.setValue(self.maxSNR)
        self.sp_snr_min.valueChanged.connect(self.on_snr_min_changed)
        self.sp_snr_max.valueChanged.connect(self.on_snr_max_changed)

        lay.addWidget(QLabel("Min. SNR (dB)"), 0, 0)
        lay.addWidget(self.sp_snr_min, 0, 1)
        lay.addWidget(QLabel("Max. SNR (dB)"), 0, 2)
        lay.addWidget(self.sp_snr_max, 0, 3)

        self.cb_awgn = QCheckBox("AWGN")
        self.cb_rayleigh = QCheckBox("Rayleigh")
        self.cb_mimo = QCheckBox("MIMO")
        self.cb_veh = QCheckBox("Vehicular")

        self.cb_awgn.setChecked(True)
        self.cb_awgn.stateChanged.connect(lambda s: self.on_channel_changed(0, s))
        self.cb_rayleigh.stateChanged.connect(lambda s: self.on_channel_changed(1, s))
        self.cb_mimo.stateChanged.connect(lambda s: self.on_channel_changed(2, s))
        self.cb_veh.stateChanged.connect(lambda s: self.on_channel_changed(3, s))

        lay.addWidget(self.cb_awgn, 1, 0)
        lay.addWidget(self.cb_rayleigh, 1, 1)
        lay.addWidget(self.cb_veh, 1, 2)
        lay.addWidget(self.cb_mimo, 1, 3)

        self.cb_n2 = QCheckBox("N = 2 antennas")
        self.cb_n4 = QCheckBox("N = 4 antennas")
        self.cb_n8 = QCheckBox("N = 8 antennas")
        self.cb_n2.stateChanged.connect(lambda s: self.on_ant_changed(0, s))
        self.cb_n4.stateChanged.connect(lambda s: self.on_ant_changed(1, s))
        self.cb_n8.stateChanged.connect(lambda s: self.on_ant_changed(2, s))

        lay.addWidget(self.cb_n2, 2, 1)
        lay.addWidget(self.cb_n4, 2, 2)
        lay.addWidget(self.cb_n8, 2, 3)
        return box

    def _build_info_group(self) -> QGroupBox:
        box = QGroupBox("Info. del Sistema")
        lay = QFormLayout(box)

        self.txt_cap = QTextEdit()
        self.txt_eb = QTextEdit()
        self.txt_dist = QTextEdit()
        self.txt_rate = QTextEdit()
        self.txt_tasa = QTextEdit()

        for t in [self.txt_cap, self.txt_eb, self.txt_dist, self.txt_rate, self.txt_tasa]:
            t.setFixedHeight(32)
            t.setReadOnly(True)

        self.btn_info = QPushButton("Info")
        self.btn_info.clicked.connect(self.on_info)
        self.lbl_info_mode = QLabel("Modo: -")

        lay.addRow("Capacidad (bits/s/Hz)", self.txt_cap)
        lay.addRow("Eb", self.txt_eb)
        lay.addRow("Distancia Minima", self.txt_dist)
        lay.addRow("Rate", self.txt_rate)
        lay.addRow("Tasa Tx (bits/s/Hz)", self.txt_tasa)
        info_row = QHBoxLayout()
        info_row.addWidget(self.btn_info)
        info_row.addWidget(self.lbl_info_mode)
        info_row.addStretch(1)
        lay.addRow(info_row)

        return box

    def _show_logo(self) -> None:
        self.canvas.ax.clear()
        logo = Path(__file__).resolve().parent / "logo.png"
        if logo.exists():
            arr = plt.imread(str(logo))
            self.canvas.ax.imshow(arr)
            self.canvas.ax.set_title("HESCOD")
            self.canvas.ax.axis("off")
        else:
            self.canvas.ax.text(0.5, 0.5, "HESCOD", ha="center", va="center", fontsize=22)
            self.canvas.ax.axis("off")
        self.canvas.draw()

    def _warn(self, msg: str) -> None:
        QMessageBox.warning(self, "Aviso", msg)

    def _parse_conv(self, text: str) -> List[int]:
        vals: List[int] = []
        for p in text.replace(",", " ").split():
            vals.append(int(p.strip()))
        return vals

    def _ensure_coding_selected(self) -> None:
        if not any(self.codingMethod):
            self.codingMethod[0] = True
            self.cb_no.blockSignals(True)
            self.cb_no.setChecked(True)
            self.cb_no.blockSignals(False)

    def _update_coding_option_widgets(self) -> None:
        is_no = bool(self.codingMethod[0])
        is_ham = bool(self.codingMethod[1])
        is_conv = bool(self.codingMethod[2])
        is_ldpc = bool(self.codingMethod[3])
        is_rs = bool(self.codingMethod[4])

        self.cb_show_const.setEnabled(is_no)
        if not is_no:
            self.cb_show_const.blockSignals(True)
            self.cb_show_const.setChecked(False)
            self.cb_show_const.blockSignals(False)
            self.show_const_received = False

        for w in [self.cb_ham_l2, self.cb_ham_l3, self.cb_ham_l4]:
            w.setEnabled(is_ham)

        self.edit_conv.setEnabled(is_conv)

        for w in [self.cb_ld_12, self.cb_ld_23, self.cb_ld_34, self.cb_ld_56]:
            w.setEnabled(is_ldpc)

        for w in [self.cb_rs_l3, self.cb_rs_l4, self.cb_rs_l5, self.edit_rs_k]:
            w.setEnabled(is_rs)

    def _update_mimo_option_widgets(self) -> None:
        is_mimo = bool(self.channelType[2])
        ant_boxes = [self.cb_n2, self.cb_n4, self.cb_n8]

        for cb in ant_boxes:
            cb.setEnabled(is_mimo)

        if is_mimo:
            return

        for i, cb in enumerate(ant_boxes):
            if cb.isChecked() or self.numAntenas[i]:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
            self.numAntenas[i] = False

    # ---------- callbacks ----------

    def on_nbits_changed(self, value: int) -> None:
        self.numBits = value

    def _parse_num_bits(self) -> Optional[int]:
        raw = self.edit_nbits.text().strip().lower()
        if not raw:
            return None
        try:
            if "e" in raw or "." in raw:
                val_f = float(raw)
                if not np.isfinite(val_f):
                    return None
                val_i = int(round(val_f))
                if abs(val_f - val_i) > 1e-9:
                    return None
            else:
                val_i = int(raw)
        except ValueError:
            return None
        if val_i <= 0 or val_i > 5_000_000:
            return None
        return val_i

    def on_generate_bits(self) -> None:
        parsed = self._parse_num_bits()
        if parsed is None:
            self._warn("Ingrese un numero valido de bits (ej: 1000000 o 1e6), entre 1 y 5000000")
            return
        self.numBits = parsed
        self.sourceBits = np.random.randint(0, 2, self.numBits, dtype=np.uint8)
        self.imagen = False
        self.image_shape = None
        self.lbl_bits_status.setText(f"Bits generados: {self.sourceBits.size:,}".replace(",", "."))

    def on_load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccione una imagen para transmitir",
            "",
            "Imagenes (*.jpeg *.jpg *.png *.tif *.tiff);;Todos (*.*)",
        )
        if not file_name:
            return
        bits, shp, y = load_image_bits(Path(file_name))
        self.sourceBits = bits
        self.image_shape = shp
        self.imagen = True
        self.lbl_bits_status.setText(f"Bits cargados desde imagen: {self.sourceBits.size:,}".replace(",", "."))
        self.canvas.ax.clear()
        self.canvas.ax.imshow(y, cmap="gray", vmin=0, vmax=255)
        self.canvas.ax.set_title("Imagen fuente (Y)")
        self.canvas.ax.axis("off")
        self.canvas.draw()

    def on_order_changed(self, value: str) -> None:
        self.order = "bin" if value == "Natural" else "gray"

    def on_mod_changed(self, idx: int, state: int) -> None:
        self.modulations[idx] = state == QT_CHECKED

    def on_show_constellation(self, idx: int) -> None:
        mods = ["pam", "pam", "pam", "psk", "psk", "psk", "qam", "qam", "qam"]
        mvals = [2, 4, 8, 4, 8, 16, 16, 64, 256]
        mod = mods[idx]
        m = mvals[idx]
        symbols = constellation_symbols(mod, m, self.order)

        ax = self.canvas.ax
        ax.clear()
        ax.plot(symbols.real, symbols.imag, "r*", markersize=10)
        for i, s in enumerate(symbols):
            ax.text(float(np.real(s)) + 0.08, float(np.imag(s)) + 0.08, f"s{i}", fontsize=9)
        ax.axhline(0.0, color="k", linewidth=1.2)
        ax.axvline(0.0, color="k", linewidth=1.2)
        lim = max(1.5, float(np.max(np.abs(np.real(symbols)))) + 1.0, float(np.max(np.abs(np.imag(symbols)))) + 1.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.grid(True)
        ax.set_title(f"Simbolos de la modulacion {m}-{mod.upper()}")
        self.canvas.draw()

    def on_code_changed(self, idx: int, state: int) -> None:
        self.codingMethod[idx] = state == QT_CHECKED
        self._ensure_coding_selected()
        self._update_coding_option_widgets()

    def _selected_coding_mode_text(self, coding: Optional[Sequence[bool]] = None) -> str:
        cm = list(coding) if coding is not None else self.codingMethod

        if cm[0]:
            return "Sin Codificacion"

        if cm[1]:
            l_vals = [2, 3, 4]
            selected = [l_vals[i] for i, v in enumerate(self.infoCoding[0]) if bool(v)]
            if len(selected) == 1:
                return f"Hamming (L={selected[0]})"
            if selected:
                return f"Hamming (L={','.join(str(x) for x in selected)})"
            return "Hamming"

        if cm[2]:
            gens = [int(x) for x in self.infoCoding[1]]
            return f"Convolucional (g={gens})"

        if cm[3]:
            rates = ["1/2", "2/3", "3/4", "5/6"]
            selected = [rates[i] for i, v in enumerate(self.infoCoding[2]) if bool(v)]
            if len(selected) == 1:
                return f"LDPC ({selected[0]})"
            if selected:
                return f"LDPC ({','.join(selected)})"
            return "LDPC"

        if cm[4]:
            l_vals = [3, 4, 5]
            selected = [l_vals[i] for i, v in enumerate(self.infoCoding[3]) if bool(v)]
            k_rs = int(self.infoCoding[4])
            if len(selected) == 1:
                return f"RS (L={selected[0]}, k={k_rs})"
            if selected:
                return f"RS (L={','.join(str(x) for x in selected)}, k={k_rs})"
            return f"RS (k={k_rs})"

        return "-"

    def on_show_const_changed(self, state: int) -> None:
        self.show_const_received = state == QT_CHECKED

    def on_hamming_l_changed(self, idx: int, state: int) -> None:
        self.infoCoding[0][idx] = state == QT_CHECKED

    def on_ldpc_rate_changed(self, idx: int, state: int) -> None:
        self.infoCoding[2][idx] = state == QT_CHECKED

    def on_rs_l_changed(self, idx: int, state: int) -> None:
        self.infoCoding[3][idx] = state == QT_CHECKED

    def on_rs_k_changed(self, value: str) -> None:
        v = value.strip()
        if not v:
            return
        try:
            self.infoCoding[4] = int(v)
        except ValueError:
            pass

    def on_conv_changed(self, value: str) -> None:
        try:
            self.infoCoding[1] = self._parse_conv(value)
        except ValueError:
            pass

    def on_snr_min_changed(self, value: int) -> None:
        self.minSNR = value

    def on_snr_max_changed(self, value: int) -> None:
        self.maxSNR = value

    def on_channel_changed(self, idx: int, state: int) -> None:
        self.channelType[idx] = state == QT_CHECKED
        if idx == 2:
            self._update_mimo_option_widgets()

    def on_ant_changed(self, idx: int, state: int) -> None:
        v = state == QT_CHECKED
        self.numAntenas[idx] = v

    def on_info(self) -> None:
        selected_codes = [i for i, v in enumerate(self.codingMethod) if bool(v)]
        coding_for_info = list(self.codingMethod)

        if len(selected_codes) == 2 and 0 in selected_codes:
            other = [i for i in selected_codes if i != 0]
            if len(other) == 1:
                coding_for_info = [False] * 5
                coding_for_info[other[0]] = True

        s1 = sum(self.modulations)
        s2 = sum(coding_for_info)
        s3 = sum(self.channelType)
        s4 = sum(self.numAntenas)

        cond1 = coding_for_info[1] and sum(self.infoCoding[0]) < 1
        cond2 = coding_for_info[2] and len(self.infoCoding[1]) == 0
        cond3 = coding_for_info[3] and sum(self.infoCoding[2]) < 1
        cond5 = coding_for_info[4] and sum(self.infoCoding[3]) != 1

        if not (s1 == 1 and s2 == 1 and s3 == 1):
            self._warn("Debe seleccionar un esquema completo (solo uno) para ver sus detalles")
            return

        if self.channelType[2] and s4 != 1:
            self._warn("Debe seleccionar UN numero de antenas para MIMO")
            return

        if cond1 or cond2 or cond3 or cond5:
            self._warn("Revisar parametros de codificacion para escoger UN unico esquema valido")
            return

        if coding_for_info[3] and not ldpc_matrices_available():
            self._warn("Falta el archivo ieee802_16e_matrices.mat para usar LDPC")
            return

        capIni, capFin = getCapacity(self.channelType, self.minSNR, self.maxSNR, self.numAntenas)
        Eb, minDist, tasa, rate, w = getParameters(
            self.modulations, coding_for_info, self.infoCoding, self.numAntenas
        )

        self.txt_cap.setPlainText(f"Min: {capIni:2.3f} b/s     Max: {capFin:2.3f} b/s")
        self.txt_eb.setPlainText(f"{Eb:2.2f} J")
        self.txt_dist.setPlainText(f"{minDist:d}")
        self.txt_tasa.setPlainText(f"{tasa:2.2f} b/s")
        self.txt_rate.setPlainText(f"{rate:2.2f}")
        self.lbl_info_mode.setText(f"Modo: {self._selected_coding_mode_text(coding_for_info)}")

        if w:
            self._warn("El codigo elegido es catastrofico")

    def _validate_before_sim(self) -> bool:
        cond1 = self.codingMethod[1] and sum(self.infoCoding[0]) < 1
        cond2 = self.codingMethod[2] and len(self.infoCoding[1]) == 0
        cond3 = self.codingMethod[3] and sum(self.infoCoding[2]) < 1
        cond5 = self.codingMethod[4] and sum(self.infoCoding[3]) < 1
        cond4 = sum(self.channelType) < 1

        if self.sourceBits.size == 0:
            self._warn("Debe generar primero el bitstream a transmitir")
            return False
        if sum(self.modulations) < 1:
            self._warn("Debe seleccionar primero algun esquema para modular la informacion")
            return False
        if cond1 or cond2 or cond3 or cond5:
            self._warn("Error en los parametros de la etapa de codificacion")
            return False
        if self.channelType[2] and sum(self.numAntenas) < 1:
            self._warn("Debe seleccionar al menos un numero de antenas para MIMO")
            return False
        if self.codingMethod[3] and not ldpc_matrices_available():
            self._warn("Falta el archivo ieee802_16e_matrices.mat para usar LDPC")
            return False
        if cond4:
            self._warn("Debe seleccionar un tipo de canal")
            return False
        if not isinstance(self.infoCoding[4], int) or self.infoCoding[4] % 2 == 0:
            self._warn("El parametro k en los codigos RS debe ser impar")
            return False
        return True

    def on_simulate(self) -> None:
        if not self._validate_before_sim():
            return

        if self._sim_thread is not None and self._sim_thread.isRunning():
            self._warn("Ya hay una simulacion en ejecucion")
            return

        snr = np.arange(self.minSNR, self.maxSNR + 1, 1, dtype=float)
        cfg = {
            "imagen": bool(self.imagen),
            "image_shape": self.image_shape,
            "sourceBits": np.array(self.sourceBits, copy=True),
            "modulations": list(self.modulations),
            "order": self.order,
            "coding": list(self.codingMethod),
            "infoCoding": [
                list(self.infoCoding[0]),
                list(self.infoCoding[1]),
                list(self.infoCoding[2]),
                list(self.infoCoding[3]),
                int(self.infoCoding[4]),
            ],
            "channelType": list(self.channelType),
            "numAntennas": list(self.numAntenas),
            "SNRdB": snr,
            "showConst": bool(self.show_const_received),
        }

        if cfg["imagen"] and cfg["image_shape"] is None:
            self._warn("No se detecto forma de imagen")
            return

        self._start_simulation(cfg)

    def _start_simulation(self, cfg: dict) -> None:
        self.btn_sim.setEnabled(False)
        self.btn_adapt.setEnabled(False)

        self._progress_dialog = QProgressDialog("Iniciando simulacion...", "Cancelar", 0, 0, self)
        self._progress_dialog.setWindowTitle("Simulando")
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(250)
        self._progress_dialog.setAutoClose(True)
        self._progress_dialog.setAutoReset(True)
        self._progress_dialog.canceled.connect(self._cancel_running_simulation)
        self._progress_dialog.open()

        self._sim_thread = QThread(self)
        self._sim_worker = SimulationWorker(cfg)
        self._sim_worker.moveToThread(self._sim_thread)

        self._sim_thread.started.connect(self._sim_worker.run)
        self._sim_worker.progress.connect(self._on_simulation_progress)
        self._sim_worker.finished.connect(self._on_simulation_finished)
        self._sim_worker.failed.connect(self._on_simulation_failed)
        self._sim_worker.cancelled.connect(self._on_simulation_cancelled)

        self._sim_worker.finished.connect(self._sim_thread.quit)
        self._sim_worker.failed.connect(self._sim_thread.quit)
        self._sim_worker.cancelled.connect(self._sim_thread.quit)
        self._sim_thread.finished.connect(self._on_thread_finished)

        self._sim_thread.start()

    def _cancel_running_simulation(self) -> None:
        if self._sim_worker is not None:
            self._sim_worker.cancel()
        if self._progress_dialog is not None:
            self._progress_dialog.setLabelText("Cancelando simulacion...")

    def _close_progress_dialog(self) -> None:
        dialog = self._progress_dialog
        if dialog is None:
            return
        self._progress_dialog = None
        try:
            dialog.canceled.disconnect(self._cancel_running_simulation)
        except Exception:
            pass
        try:
            dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass

    def _on_simulation_progress(self, done: int, total: int, message: str) -> None:
        dialog = self._progress_dialog
        if dialog is None:
            return
        safe_total = max(1, int(total))
        safe_done = max(0, min(int(done), safe_total))
        try:
            if dialog.maximum() <= 0:
                dialog.setRange(0, safe_total)
            else:
                dialog.setMaximum(safe_total)
            dialog.setValue(safe_done)
            dialog.setLabelText(message)
        except (RuntimeError, AttributeError):
            return

    def _on_simulation_finished(self, results: Sequence[dict], is_image: bool) -> None:
        dialog = self._progress_dialog
        if dialog is not None:
            try:
                dialog.setLabelText("Finalizando...")
                dialog.setValue(dialog.maximum())
            except RuntimeError:
                pass
        self._close_progress_dialog()

        self.btn_sim.setEnabled(True)
        self.btn_adapt.setEnabled(True)

        if is_image:
            self._plot_image_results(results)
            return

        self._plot_ber_results(results)
        if self.show_const_received:
            for item in results:
                if item["params"].codeType == "NoCoding":
                    self._plot_received_constellations(
                        item["modulated_hist"],
                        item["received_hist"],
                        item["symbols"],
                        item["snr"],
                        item["params"].mod,
                        item["params"].niveles,
                    )

    def _on_simulation_failed(self, error_message: str) -> None:
        self._close_progress_dialog()

        self.btn_sim.setEnabled(True)
        self.btn_adapt.setEnabled(True)
        self._warn(f"Error durante la simulacion: {error_message}")

    def _on_simulation_cancelled(self) -> None:
        self._close_progress_dialog()

        self.btn_sim.setEnabled(True)
        self.btn_adapt.setEnabled(True)

    def _on_thread_finished(self) -> None:
        self._sim_worker = None
        self._sim_thread = None

    def _plot_ber_results(self, results: Sequence[dict]) -> None:
        fig, ax = plt.subplots(figsize=(11, 7))
        legends = []
        for item in results:
            ax.semilogy(item["snr"], item["ber"], linewidth=2)
            legends.append(item["legend"])

        ax.set_xlabel("SNR (dB)", fontsize=12)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_title("Curvas BER", fontsize=14, fontweight="bold")
        ax.grid(True)
        if legends:
            ax.legend(legends)
        fig.tight_layout()
        fig.show()

    def _plot_received_constellations(
        self,
        modulated_hist: Sequence[np.ndarray],
        received_hist: Sequence[np.ndarray],
        symbols: np.ndarray,
        snr_vals: Sequence[float],
        mod: str,
        niveles: int,
    ) -> None:
        if len(modulated_hist) == 0:
            return
        pos = np.linspace(0, len(modulated_hist) - 1, 4, dtype=int)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        colors = np.minimum(0.9, 0.2 + np.random.rand(niveles, 3))

        for ax, p in zip(axes.ravel(), pos):
            m = modulated_hist[p]
            r = received_hist[p]
            ax.set_title(f"{niveles}-{mod.upper()} -- SNR = {snr_vals[p]:g} dB")
            limit = 1.0

            for j, s in enumerate(symbols):
                rj = r[m == s]
                if rj.size == 0:
                    continue
                if np.isrealobj(rj):
                    ax.plot(rj.real, np.zeros_like(rj.real), "*", color=colors[j], linestyle="None")
                else:
                    ax.plot(rj.real, rj.imag, "*", color=colors[j], linestyle="None")
                limit = max(limit, float(np.max(np.abs(np.real(rj)))), float(np.max(np.abs(np.imag(rj)))))

            ax.plot(symbols.real, symbols.imag, "k*", markersize=8)
            lim = np.ceil(limit) + 1
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])
            ax.grid(True)

            if mod == "psk":
                angles = np.linspace(0, 2 * np.pi, niveles, endpoint=False) + np.pi / niveles
                for ang in angles:
                    ax.plot([0, 5 * np.cos(ang)], [0, 5 * np.sin(ang)], color=(1.0, 0.5, 0.0), linewidth=1.5)
            else:
                mx = max(1.0, float(np.max(np.abs(symbols.real))))
                for x in np.arange(-mx + 1, mx + 0.1, 2):
                    ax.axvline(x=x, color=(1.0, 0.5, 0.0), linewidth=1.2)
                if mod == "qam":
                    for y in np.arange(-mx + 1, mx + 0.1, 2):
                        ax.axhline(y=y, color=(1.0, 0.5, 0.0), linewidth=1.2)

        fig.suptitle("Constelacion de Simbolos Recibidos", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.show()

    def _plot_image_results(self, results: Sequence[dict]) -> None:
        for item in results:
            p = item["params"]
            snr = item["snr"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"Transmision de la imagen fuente usando {p.niveles}-{p.mod.upper()} y SNR = {snr:g} dB",
                fontsize=13,
            )
            axes[0].imshow(item["tx_image"], cmap="gray", vmin=0, vmax=255)
            axes[0].set_title("Imagen Original")
            axes[0].axis("off")
            axes[1].imshow(item["rx_image"], cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Imagen Recibida")
            axes[1].axis("off")
            fig.tight_layout()
            fig.show()

    def on_plot_adapt(self) -> None:
        base = Path(__file__).resolve().parent
        candidates = [base / "datosAdapt.mat", base.parent / "datosAdapt.mat"]
        mat_path = next((p for p in candidates if p.exists()), candidates[0])
        if not mat_path.exists():
            self._warn("No se encuentra datosAdapt.mat")
            return

        data = loadmat(mat_path)
        ind = np.array(data["ind"]).reshape(-1)
        ber = np.array(data["ber"])
        rates_raw = data["rates"].reshape(-1)
        rates: List[str] = []
        for r in rates_raw:
            if isinstance(r, np.ndarray):
                rv = r.reshape(-1)
                rates.append(str(rv[0]) if rv.size else "?")
            else:
                rates.append(str(r))
        num_niveles = np.array(data["numNiveles"]).reshape(-1).astype(int)

        fig, ax = plt.subplots(figsize=(11, 7))
        lines = ax.plot(ind, ber.T, linewidth=2)
        ax.set_ylabel("BER", fontsize=12)
        ax.set_xlabel("Tasa Tx. (bits/s/Hz)", fontsize=12)
        ax.grid(True)

        snr_labels = ["0 dB", "2 dB", "5 dB", "8 dB", "10 dB", "15 dB"]
        curve_labels = snr_labels[: len(lines)]
        if len(curve_labels) < len(lines):
            for i in range(len(curve_labels), len(lines)):
                curve_labels.append(f"Curva {i + 1}")
        ax.legend(lines, curve_labels)

        xlabels: List[str] = []
        for x, m, r in zip(ind, num_niveles, rates):
            mod = "QAM" if m >= 9 else "PSK"
            xlabels.append(f"{x:g}\n{m}-{mod}\nLDPC {r}")

        if ind.size == len(xlabels):
            ax.set_xticks(ind)
            ax.set_xticklabels(xlabels, fontsize=8)

        ax.set_title("Codificacion Adaptativa", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.show()


__all__ = ["MainWindow"]
