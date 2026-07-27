import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
import pyqtgraph
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QLabel, QMessageBox, \
    QVBoxLayout, QMenu, QAction, QFileDialog, QInputDialog, QColorDialog
from PyQt5.uic import loadUi
from pyqtgraph import mkPen
from scipy.signal import savgol_filter

from control.config_tree_interface import ConfigTreeInterface
from control.imp_tube_params_setting_interface import ImptubeParamsSetInterface
from control.log_manager import LogManager
from control.mic_adjust_interface import MicAdjustInterface
from control.output_signal_setting_interface import OutputSignalSetInterface
from control.output_voltage_interface import MicoutputVoltageInterface, SoundcardCalibrationManager
from control.select_deivce_interface import SelectDeviceInterface
from control.utils import utils
from control.utils.audio_session_manager import AudioSessionManager
from control.utils.streaming_audio_processor import DuplexStreamingPlayRec
from custom.customSignals import sign
from resources import icons_rc, ui_rc # 导入资源


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device_window = None
        self.imptube_window = None
        self.output_window = None
        self.mic_window = None
        self.output_voltage_window = None
        self.config_tree_window = None
        self.signal_info = None
        self.tube_params = None
        self.test_result = None
        self.soft_value = None
        self.plot_state = False
        self.mic1_data = None
        self.mic2_data = None
        self.mic1_pa = None
        self.mic2_pa = None
        self.logger = LogManager.set_log_handler("主窗口")
        # ===== 流式录音相关 =====
        self.streaming_buffer = []
        self.stream_instance = None
        self.mic_binding = (0, 1)
        self.mic_deviation_db = (0.0, 0.0)
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._handle_queue_and_update_ui)

        self.init_ui()
        self.init_fun()
        self.init_image()
        self.init_view()
        self.load_basic_params_config()
        self.init_slider()
        self.history_line = []
        self.crosshair_enabled = False

    def init_ui(self):
        ui_file = QFile(":ui/window.ui")
        ui_file.open(QFile.ReadOnly)
        loadUi(ui_file, self)
        # 设置窗口标题和大小
        self.setWindowTitle('小孔膜片阻抗管测试系统')
        self.showMaximized()

        # 设置窗口图标
        self.setWindowIcon(QIcon(':/images/dongyuan.png'))
        # 获取主界面的布局并设置内容的边距
        layout = self.centralWidget().layout()  # 获取 QMainWindow 的中心控件的布局
        if layout:
            layout.setContentsMargins(1, 1, 1, 1)

    def init_fun(self):
        self.action_2.triggered.disconnect()
        self.action_2.triggered.connect(self.open_select_device_interface)
        self.action.triggered.disconnect()
        self.action.triggered.connect(self.open_imptube_params_setting_interface)
        self.action_3.triggered.disconnect()
        self.action_3.triggered.connect(self.open_output_signal_setting_interface)
        self.action_9.triggered.disconnect()
        self.action_9.triggered.connect(self.open_mic_adjust_interface)
        self.action_6.triggered.disconnect()
        self.action_6.triggered.connect(self.open_output_voltage_interface)
        self.action_8.triggered.disconnect()
        self.action_8.triggered.connect(self.open_config_tree_interface)
        self.action_14.triggered.disconnect()
        self.action_14.triggered.connect(self.popup_pdf)
        self.run_test_button.clicked.connect(self.run_test)
        self.save_test_data.triggered.disconnect()
        self.save_test_data.triggered.connect(self.save_test_data_to_excel)
        self.plot_type_selector.currentIndexChanged.connect(self.update_plot3_by_selector)
        sign.update_plot3_by_selector_sign.connect(self.update_plot3_by_selector)

    def only_view_all_menu(self):
        def context_menu(event):
            menu = QMenu()

            # 添加 View All 操作（等价于原生行为）
            view_all_action = QAction("View All")
            menu.addAction(view_all_action)

            # 设置触发行为：缩放视图范围以适应所有数据
            view_all_action.triggered.connect(lambda: self.plot3.plotItem.enableAutoRange())

            menu.exec_(event.screenPos())

        self.plot3.plotItem.scene().contextMenuEvent = context_menu

    def init_image(self):
        scene = QGraphicsScene()
        pixmap = QPixmap(":/images/2mic薄层_画板.png")
        scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 缩小图像
        pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
        scene.addItem(pixmap_item)
        self.schematic1.setScene(scene)
        # 获取主界面的布局并设置内容的边距
        layout = self.centralWidget().layout()  # 获取 QMainWindow 的中心控件的布局
        if layout:
            layout.setContentsMargins(1, 1, 1, 1)

    def init_view(self):
        self.plot1 = pyqtgraph.PlotWidget()
        layout1 = QVBoxLayout()
        layout1.setContentsMargins(0, 0, 0, 0)
        layout1.addWidget(self.plot1)
        self.frame.setLayout(layout1)
        self.plot1.setBackground('white')
        self.plot1.setLabel('bottom', 'Time', units='s')
        self.plot1.setLabel('left', 'Amplitude', units='Pa')
        self.plot1.getAxis('bottom').setTextPen("black")
        self.plot1.getAxis('left').setTextPen("black")
        self.plot1.showGrid(x=True, y=True, alpha=0.25)
        self.curve1 = self.plot1.plot([], [], pen=mkPen(color=(76, 120, 168)))

        self.plot2 = pyqtgraph.PlotWidget()
        layout2 = QVBoxLayout()
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.addWidget(self.plot2)
        self.frame_2.setLayout(layout2)
        self.plot2.setBackground('white')
        self.plot2.setLabel('bottom', 'Time', units='s')
        self.plot2.setLabel('left', 'Amplitude', units='Pa')
        self.plot2.getAxis('bottom').setTextPen("black")
        self.plot2.getAxis('left').setTextPen("black")
        self.plot2.showGrid(x=True, y=True, alpha=0.25)
        self.curve2 = self.plot2.plot([], [], pen=mkPen(color=(84, 162, 75)))

        self.plot3 = pyqtgraph.PlotWidget()
        layout3 = QVBoxLayout()
        layout3.setContentsMargins(0, 0, 0, 0)
        layout3.addWidget(self.plot3)
        self.frame_3.setLayout(layout3)
        self.plot3.setBackground('white')
        self.plot3.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plot3.customContextMenuRequested.connect(self.on_plot3_menu)

        self.plot3.getAxis('bottom').enableAutoSIPrefix(False)
        self.plot3.getAxis('left').enableAutoSIPrefix(False) # 禁用自动转换单位
        self.plot3.getAxis('bottom').setLogMode(True)  # 对数坐标轴，适用于频率
        self.plot3.getAxis('left').setLogMode(True)
        # 使用自定义的对数坐标轴刻度标签格式
        self.plot3.getAxis('bottom').logTickStrings = utils.custom_log_tick_strings
        # self.plot3.getAxis('bottom').setTickSpacing(levels=[(1, 1)])
        # 设置网格线和背景
        self.plot3.getPlotItem().showGrid(x=True, y=True)

        # 调整刻度字体大小
        font = pyqtgraph.QtGui.QFont()
        font.setPointSize(12)
        self.plot3.getAxis('bottom').setTickFont(font)
        self.plot3.getAxis('left').setTickFont(font)
        self.plot3.setLabel('left', 'Z abs', units='Rayl')
        self.plot3.setLabel('bottom', 'Freq', units='Hz')
        self.plot3.getAxis('bottom').setTextPen("black")
        self.plot3.getAxis('left').setTextPen("black")
        # 清空范围限制，允许滚轮缩放
        self.plot3.setAutoVisible(True)  # 启用自动范围调整
        self.plot3.plotItem.scene().sigMouseMoved.connect(self.mov)
        # legend 只创建一次
        self.legend = self.plot3.addLegend(offset=(-10, 10))
        # 主曲线（只建一次）
        self.curve3 = self.plot3.plot([], [], pen=mkPen(color='black', width=2), name='本次测试')

    def on_plot3_clicked(self, event):
        if event.double():
            mouse_point = self.plot3.getViewBox().mapSceneToView(event.scenePos())
            x_log_clicked = mouse_point.x()
            freq_clicked = 10 ** x_log_clicked  # 实际频率

            plot_type = self.plot_type_selector.currentText()

            def safe_log(a):
                a = np.asarray(a, dtype=float)
                a[a <= 0] = np.nan  # 防止取对数报错
                return np.log10(a)

            if plot_type == "传输阻抗率Z abs":
                # 获取当前显示的曲线数据
                freq_array = np.asarray(self.test_result["f"], dtype=float)
                y_array = self.test_result["Z_abs"]
            elif plot_type == "传输阻抗率Z Re":
                freq_array = np.asarray(self.test_result["f"], dtype=float)
                y_array = np.abs(self.test_result["Z_Re"])
            elif plot_type == "传输阻抗率Z Im":
                freq_array = np.asarray(self.test_result["f"], dtype=float)
                y_array = np.abs(self.test_result["Z_Im"])
            else:
                return
            y_disp = safe_log(y_array)
            x_disp = np.log10(freq_array)
            # 如果启用了平滑处理
            if self.soft_value > 3:
                y_disp = savgol_filter(y_disp, window_length=self.soft_value, polyorder=3)
            # 找出频率中最接近点击频率的点
            index = np.abs(freq_array - freq_clicked).argmin()
            x_log = float(x_disp[index])  # 显示坐标（log10 频率）
            y_log = float(y_disp[index])  # 显示坐标（log10 幅值）
            x_lin = float(freq_array[index])  # 线性频率x
            y_lin = float(10 ** y_disp[index]) # 线性幅值y, 根据对数平滑生成的,而非原始数据

            self.logger.info(f"双击事件: 频率(x): {freq_clicked:.2f} Hz, 最近数据索引: {index}")
            self.logger.info(f"对数频率位置: {x_log:.4f}), 对数幅值位置: {y_log:.4f}")
            self.logger.info(f"频率线性值: {x_lin:.2f} Hz, 幅值线性值: {y_lin:.4f}")
            # 初始化十字线和文字提示
            if not self.crosshair_enabled:
                self.vLine = pyqtgraph.InfiniteLine(angle=90, movable=False, pen='r')
                self.hLine = pyqtgraph.InfiniteLine(angle=0, movable=False, pen='r')
                self.text = pyqtgraph.TextItem("", anchor=(0, 1), fill=pyqtgraph.mkBrush(255, 255, 255, 200),
                                               border='k')
                self.plot3.addItem(self.vLine, ignoreBounds=True)
                self.plot3.addItem(self.hLine, ignoreBounds=True)
                self.plot3.addItem(self.text)
                self.plot3.plotItem.scene().sigMouseMoved.connect(self.mov)
                self.crosshair_enabled = True

            self.vLine.setPos(x_log)
            self.hLine.setPos(y_log)

            # 添加点击标记
            if getattr(self, "click_marker", None) is not None:
                self.plot3.removeItem(self.click_marker)
            self.click_marker = pyqtgraph.ScatterPlotItem(
                [x_log], [y_log],
                symbol='o', size=4, brush=pyqtgraph.mkBrush(255, 255, 0), pen='k'
            )
            self.plot3.addItem(self.click_marker)

            # 提示框（显示线性频率和与图一致的线性幅值）
            self.text.setHtml(
                f"<div style='background-color:white; padding:2px;'>"
                f"<b>频率(x):</b> {x_lin:.2f} Hz<br>"
                f"<b>幅值(y):</b> {y_lin:.4f}</div>"
            )
            self.text.setPos(x_log, y_log)

    def on_plot3_menu(self, pos):
        menu = QMenu(self)

        view_all_action = QAction("View All", self)
        view_all_action.triggered.connect(lambda: self.plot3.plotItem.enableAutoRange())
        menu.addAction(view_all_action)

        save_action = QAction("保存当前曲线", self)
        save_action.triggered.connect(self.save_test_result_to_json)
        menu.addAction(save_action)

        menu.exec_(self.plot3.mapToGlobal(pos))

    def save_test_result_to_json(self):
        if not self.test_result:
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return

        default_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name, ok = QInputDialog.getText(self, "保存历史线名称", "请输入名称：", text=default_name)
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, "提示", "名称不能为空！")
            return

        color = QColorDialog.getColor(parent=self, title="选择颜色")
        if not color.isValid():
            return
        colour_str = color.name()

        content = self._read_tree_config()
        line_cfg = self._ensure_tree_line_config(content)
        new_index = len(line_cfg["manage_history_line"])
        if new_index >= 10:
            QMessageBox.warning(self, "提示", "历史数据已达 10 条，请先删除后再保存。")
            return

        line_cfg["saved_history_line"].append(self._json_safe(self.test_result))
        line_cfg["manage_history_line"][str(new_index)] = {
            "state": "True",
            "colour": colour_str,
            "line_name": name,
        }

        if utils.write_config_content("tree_config.json", content):
            QMessageBox.information(self, "成功", f"已保存历史线：{name}\n颜色：{colour_str}")
            if self.config_tree_window and self.config_tree_window.isVisible():
                self.config_tree_window.fill_page_2()
            self.update_plot3_by_selector()
            QApplication.processEvents()
        else:
            QMessageBox.critical(self, "错误", "写入 tree_config.json 失败，请重试。")

    @staticmethod
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: MainWindow._json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [MainWindow._json_safe(val) for val in value]
        return value

    @staticmethod
    def _default_tree_config():
        return {
            "line": {
                "is_display_history_line": 1,
                "manage_history_line": {},
                "saved_history_line": [],
            }
        }

    def _read_tree_config(self):
        res, content = utils.get_config_content("tree_config.json")
        if not res or not isinstance(content, dict):
            content = self._default_tree_config()
        self._ensure_tree_line_config(content)
        return content

    def _ensure_tree_line_config(self, content):
        line_cfg = content.setdefault("line", {})
        line_cfg.setdefault("is_display_history_line", 1)
        line_cfg.setdefault("manage_history_line", {})
        line_cfg.setdefault("saved_history_line", [])
        return line_cfg

    def _get_plot3_display_data(self, result):
        text = self.plot_type_selector.currentText()
        f = np.asarray(result["f"], dtype=float)

        if "abs" in text:
            y = np.asarray(result["Z_abs"], dtype=float)
        elif "Re" in text:
            y = np.abs(np.asarray(result["Z_Re"], dtype=float))
        elif "Im" in text:
            y = np.abs(np.asarray(result["Z_Im"], dtype=float))
        else:
            raise ValueError(f"未知曲线类型：{text}")

        log_f = np.log10(f)
        y = y.copy()
        y[y <= 0] = np.nan
        log_y = np.log10(y)
        if self.soft_value > 3:
            log_y = savgol_filter(log_y, window_length=self.soft_value, polyorder=3)
        return log_f, log_y

    def _clear_history_lines(self):
        for line in getattr(self, "history_line", []):
            try:
                self.legend.removeItem(line)
            except Exception:
                pass
            try:
                self.plot3.removeItem(line)
            except Exception:
                pass
        self.history_line = []

    def update_history_lines(self):
        content = self._read_tree_config()
        line_cfg = self._ensure_tree_line_config(content)
        if int(line_cfg.get("is_display_history_line", 1)) == 0:
            return

        saved_lines = line_cfg.get("saved_history_line", [])
        id_list = []
        for key, cfg in line_cfg.get("manage_history_line", {}).items():
            if str(cfg.get("state", "False")) != "True":
                continue
            try:
                index = int(key)
            except Exception:
                self.logger.warning(f"历史线索引无法转换为 int: {key}")
                continue
            if 0 <= index < len(saved_lines):
                id_list.append((index, cfg.get("colour", "#1f77b4"), cfg.get("line_name", f"历史线{index}")))
        id_list.sort()

        for index, colour, line_name in id_list:
            self.add_history_line(colour, saved_lines[index], line_name)

    def add_history_line(self, colour, original_data, line_name):
        try:
            log_f, log_y = self._get_plot3_display_data(original_data)
        except Exception as exc:
            self.logger.warning(f"历史线 {line_name} 数据无法绘制: {exc}")
            return

        item = self.plot3.plot(log_f, log_y, pen=mkPen(color=colour, width=1), name=line_name)
        self.history_line.append(item)

    def mov(self, pos):
        if self.plot3.sceneBoundingRect().contains(pos):
            mouse_point = self.plot3.getViewBox().mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()
            real_x = 10 ** x
            real_y = 10 ** y
            threshold = 1e10
            if real_x > threshold:
                x_str = "∞"
            else:
                x_str = f"{real_x:.2f}"
            if real_y > threshold:
                y_str = "∞"
            else:
                y_str = f"{real_y:.2f}"
                self.common_pos_value.setText(f"({x_str}, {y_str})")

    def init_slider(self):
        self.slider = self.findChild(QtWidgets.QSlider, "slider")
        # 添加浮动标签
        self.float_label = QLabel(self)
        self.float_label.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.float_label.setFixedSize(60, 20)
        self.float_label.setAlignment(Qt.AlignCenter)
        self.float_label.show()
        self.float_label.raise_()

        # 初始化浮动标签
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(10)
        self.slider.setValue(self.soft_value)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

        self.slider.valueChanged.connect(self.update_float_label)

    def _reposition_slider_label(self):
        if hasattr(self, "slider") and self.slider is not None:
            self.update_float_label(self.slider.value())

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._reposition_slider_label)  # 首次显示后再定位

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reposition_slider_label)  # 窗口尺寸变化后再定位

    def update_float_label(self, value):
        float_value = value / 1
        self.float_label.setText(f"{float_value:.1f}")
        self.float_label.setStyleSheet("""
            background-color: #f5f5f5;
            border: 1px solid #999;
            border-radius: 4px;
            padding: 2px;
            font-size: 10pt;
        """)
        # 获取 slider 左上角全局位置 → 相对于 self 的偏移
        slider_pos = self.slider.mapTo(self, self.slider.rect().topLeft())
        slider_width = self.slider.width()
        value_ratio = (value - self.slider.minimum()) / (self.slider.maximum() - self.slider.minimum())

        # 更精确地估计 handle 的像素位置（减去10调整）
        handle_x = int(value_ratio * (slider_width - 20))

        # 设置浮动标签的位置（居中显示）
        self.float_label.move(
            slider_pos.x() + handle_x - self.float_label.width() // 2 + 10,
            slider_pos.y() - self.float_label.height() - 5
        )
        soft_value = int(float(self.float_label.text()))
        self.save_basic_params_config(soft_value)
        self.load_basic_params_config()
        if self.plot_state is True:
            self.update_plot3_by_selector()
        else:
            # 不更新界面
            pass

    def load_basic_params_config(self):
        basic_params_path = utils.get_config_path("basic_params.json")
        try:
            with open(basic_params_path , "r", encoding="utf-8") as f:
                basic_config  = json.load(f)
                self.soft_value = basic_config.get("basic_params").get("soft_value")
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{basic_params_path}配置失败：{e}")

    def save_basic_params_config(self, soft_value=None, **kwargs):
        basic_params_path = utils.get_config_path("basic_params.json")
        try:
            with open(basic_params_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "basic_params" not in config:
                config["basic_params"] = {}

            config["basic_params"]["soft_value"] = soft_value

            with open(basic_params_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"写入配置失败：{e}")


    def move_to_center(self):
        desktop = QApplication.desktop().availableGeometry()
        w, h = desktop.width(), desktop.height()
        # self.resize(w, h)
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)

    def open_select_device_interface(self):
        if self.device_window is None:
            self.device_window = SelectDeviceInterface(self)
        self.device_window.show()
        self.device_window.raise_()  # 窗口置顶
        self.device_window.activateWindow()


    def open_imptube_params_setting_interface(self):
        if self.imptube_window is None:
            self.imptube_window = ImptubeParamsSetInterface(self)
        self.imptube_window.show()
        self.imptube_window.raise_()  # 窗口置顶
        self.imptube_window.activateWindow()

    def open_output_signal_setting_interface(self):
        if self.output_window is None:
            self.output_window = OutputSignalSetInterface(self)
        self.output_window.show()
        self.output_window.raise_()
        self.output_window.activateWindow()

    def open_mic_adjust_interface(self):
        if self.mic_window is None:
            self.mic_window = MicAdjustInterface(self)
        self.mic_window.show()
        self.mic_window.raise_()
        self.mic_window.activateWindow()

    def open_output_voltage_interface(self):
        if self.output_voltage_window is None:
            self.output_voltage_window = MicoutputVoltageInterface(self)
        self.output_voltage_window.show()
        self.output_voltage_window.raise_()
        self.output_voltage_window.activateWindow()

    def open_config_tree_interface(self):
        if self.config_tree_window is None:
            self.config_tree_window = ConfigTreeInterface(self)
        self.config_tree_window.show()
        self.config_tree_window.raise_()
        self.config_tree_window.activateWindow()

    def init_config(self):
        output_path = utils.get_config_path("output_signal_setting.json")
        try:
            with open(output_path , "r", encoding="utf-8") as f:
                output_config  = json.load(f)
                self.signal_info = output_config .get("signal_info", {})
        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{output_path}配置失败：{e}")

        tube_path = utils.get_config_path("imp_tube_params_setting.json")
        try:
            with open(tube_path , "r", encoding="utf-8") as f:
                tube_config  = json.load(f)
                self.tube_params = tube_config.get("tube_params", {})

        except Exception as e:
            self.logger.error(f"读取配置失败: {e}")
            QMessageBox.warning(self, "错误", f"读取{tube_path}配置失败：{e}")

    def _get_lumped_parameter_inputs(self):
        if not self.tube_params:
            QMessageBox.warning(self, "参数错误", "阻抗管参数未加载，请先检查参数设置。")
            return None
        try:
            tube_temperature = float(self.tube_params.get("tube_temperature"))
            s_sample_mm2 = float(self.tube_params.get("s_sample_mm2"))
            v_backing_cc = float(self.tube_params.get("v_backing_cc"))
        except (TypeError, ValueError):
            QMessageBox.warning(self, "参数错误", "阻抗管参数不完整，请填写管中温度、待测样品面积和背腔体积。")
            return None

        if s_sample_mm2 <= 0:
            QMessageBox.warning(self, "参数错误", "待测样品面积 s_sample_mm2 必须大于 0。")
            return None
        if v_backing_cc <= 0:
            QMessageBox.warning(self, "参数错误", "背腔体积 v_backing_cc 必须大于 0。")
            return None

        return tube_temperature, s_sample_mm2, v_backing_cc

    def _load_mic_binding_indices(self):
        # 默认顺序
        idx1, idx2 = 0, 1
        dev1, dev2 = 0.0, 0.0
        try:
            path = utils.get_config_path("mic_calibration.json")
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            idx1 = int(cfg.get("In1", {}).get("binding_index", idx1))
            idx2 = int(cfg.get("In2", {}).get("binding_index", idx2))
            dev1 = float(cfg.get("In1", {}).get("deviation_value", dev1))
            dev2 = float(cfg.get("In2", {}).get("deviation_value", dev2))
        except Exception as e:
            self.logger.warning(f"读取 mic_calibration.json 失败，使用默认绑定 0/1：{e}")

        self.mic_binding = (idx1, idx2)
        self.mic_deviation_db = (dev1, dev2)

    def run_test(self):
        self.init_config()
        self._load_mic_binding_indices()
        # 防止多个界面同时占用声卡
        if not AudioSessionManager.acquire(self):
            QMessageBox.warning(self, "提示", "音频设备正在被其它界面占用，请先停止/关闭其它录音功能。")
            self.test_state.setText("音频被占用")
            return
        utils.set_run_button_enabled(self.run_test_button, False)
        self.test_state.setText("测试中...")
        QApplication.processEvents()
        self.record_and_plot()

    def clear_plot3_result(self):
        self.curve3.setData([], [])
        self._clear_history_lines()
        if self.crosshair_enabled:
            self.del_cross()
        self.test_result = None
        self.plot_state = False
        self.crosshair_enabled = False
        self.vLine = None
        self.hLine = None
        self.text = None
        self.click_marker = None

    def record_and_plot(self):
        try:
            # 清空显示（不要 plot().clear() 叠加）
            self.curve1.setData([], [])
            self.curve2.setData([], [])
            self.clear_plot3_result()
            # 清除 十字线
            if self.crosshair_enabled:
                self.del_cross()
            self.streaming_buffer = []
            QApplication.processEvents()
            # 获取设备采样率
            samplerate = utils.get_device_info()
            duration = self.signal_info['signal_time']
            # 根据目标电压，校准并生成激励信号
            result, data, amplitude, msg = utils.generate_calibrated_signal(self.signal_info, samplerate)
            if result is False:
                self.test_state.setText("缺少输出校准文件")
                QMessageBox.warning(self, "提示", "缺少输出校准文件，请先进行输入/输出校准。")
                utils.set_run_button_enabled(self.run_test_button, True)
                AudioSessionManager.release(self)
                return

            input_channels = sd.query_devices(sd.default.device[0], 'input')['max_input_channels']
            output_channels = sd.query_devices(sd.default.device[1], 'output')['max_output_channels']
            print(f"最大输入通道数: {input_channels},输出通道: {output_channels}")
            self.logger.info(f"最大输入通道数: {input_channels},输出通道: {output_channels}")

            if input_channels < 2:
                raise ValueError(f"检测到的输入通道数为 {input_channels},不足 2 个，无法进行完整校准，请选择别的设备！")

            # 开始录音 (多声道)
            stimulus_data = np.asarray(data, dtype=np.float32)
            if output_channels > 1:
                stimulus_data = np.column_stack([stimulus_data] * output_channels)

            # 读设备（你已经保存到 basic_params.json）
            ok, config = utils.get_config_content("basic_params.json")
            input_device = config["basic_params"]["input_selected_device_id"]
            output_device = config["basic_params"]["output_selected_device_id"]
            self.logger.info(f"开始流式播放并录音..")

            self.stream_instance = DuplexStreamingPlayRec()
            self.stream_instance.start(
                stimulus_data=stimulus_data,
                sample_rate=samplerate,
                input_device=input_device,
                output_device=output_device,
                input_channels=int(input_channels),
                output_channels=int(output_channels),
                duration=duration,  # 你没有 prepare/prolong 就用 duration
                blocksize=2048,
            )
            self.stream_timer.start(50)  # 50ms 刷一次队列/判断结束
            return

        except Exception as e:
            self.test_state.setText(f"测试失败: {e}")
            QMessageBox.critical(self, "错误", f"录音失败：{e}")
            utils.set_run_button_enabled(self.run_test_button, True)
            AudioSessionManager.release(self)

    def _handle_queue_and_update_ui(self):
        """
        仿 SpeakerAnomalyDetection 的写法：
        - 定时器里只做三件事：处理队列、实时更新显示、判断结束并触发收尾
        """
        if self.stream_instance is None:
            return

        # 1) 处理队列：把回调线程塞进来的 chunk 取出来
        chunks = self.stream_instance.process_queue()
        sr = self.stream_instance.sample_rate

        if chunks:
            # 2) 实时显示：只画最近 N 秒（避免越画越卡）
            duration = float(self.signal_info.get("signal_time", 5))
            keep_sr = int(duration * sr)

            # 维护滚动窗口（只保留最近 keep 个采样点）
            new_block = np.vstack(chunks)  # (k,4)
            self.streaming_buffer.append(new_block)  # 用 list 装块更快
            buf = np.vstack(self.streaming_buffer)
            if buf.shape[0] > keep_sr:
                buf = buf[-keep_sr:, :]
                self.streaming_buffer = [buf]  # 只保留一块，避免 list 无限增长

            # 时间轴：用累计样本数推算起点
            total_samples = self.stream_instance.samples_captured
            start_sample = max(0, total_samples - buf.shape[0])
            time_axis = (start_sample + np.arange(buf.shape[0])) / sr

            # 只做“时域实时更新”
            pa1, pa2 = MicAdjustInterface.process_chunks(buf, self.mic_binding, self.mic_deviation_db)
            self.update_time_plot(time_axis, pa1, pa2)

        #3) 录音结束：录音停止-> stream停timer-> 收集结果(recording/err)-> stop/释放stream-> (FFT/保存/画图)-> 恢复按钮&释放占用锁
        if not self.stream_instance.is_recording:

            self.stream_timer.stop()

            err = getattr(self.stream_instance, "error", None)
            recording = self.stream_instance.get_recorded_data()  # (N,4)

            self.stream_instance.stop()
            self.stream_instance = None

            try:
                if err:
                    raise RuntimeError(f"流式录音出错：{err}")

                self._handle_recording_fft(recording, sr)

            except Exception as e:
                self.logger.exception("录音后处理失败")
                sign.error_message_signal.emit(f"录音失败：{e}", self)

            finally:
                utils.set_run_button_enabled(self.run_test_button, True)
                AudioSessionManager.release(self)

    def _handle_recording_fft(self, recording, samplerate):
        self.logger.info("录音完成(流式),开始处理音频数据")
        duration = float(self.signal_info.get("signal_time", 0))

        # 1) 计算（SPL/scale）
        self.mic1_data, self.mic2_data, real_spl1, real_spl2, scale1, scale2 = (
            MicAdjustInterface.process_mic_channels_data(recording, self.mic_binding, self.mic_deviation_db))

        # 2) 转成 Pa
        self.mic1_pa = self.mic1_data * scale1
        self.mic2_pa = self.mic2_data * scale2

        # 3) 饱和提示
        if real_spl1 > 120:
            self.logger.error(f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风1实际声压级 {real_spl1:.2f} dB, 超过阈值 {120} dB，已饱和")
        if real_spl2 > 120:
            self.logger.error(f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")
            QMessageBox.warning(self, "警告", f"麦克风2实际声压级 {real_spl2:.2f} dB, 超过阈值 {120} dB，已饱和")


        # 4) 保存 txt（保存的是 mic*_data 原始通道数据）
        save_dir = os.path.join(os.getcwd(), "！采集")
        os.makedirs(save_dir, exist_ok=True)
        mic1_txt = os.path.join(save_dir, "MIC1.txt")
        mic2_txt = os.path.join(save_dir, "MIC2.txt")
        np.savetxt(mic1_txt, self.mic1_data)
        np.savetxt(mic2_txt, self.mic2_data)
        self.logger.info(f"MIC1 采集已保存: {mic1_txt}")
        self.logger.info(f"MIC2 采集已保存: {mic2_txt}")

        # 画最终 Pa 曲线
        time_axis = np.linspace(0, duration, len(self.mic1_pa))
        self.update_time_plot(time_axis, self.mic1_pa, self.mic2_pa)

        try:
            mic1_cal_data, mic2_cal_data = self.load_calibration_data()
        except Exception:
            self.logger.warning("缺少麦克风校准文件")
            self.test_state.setText("缺少校准文件")
            QMessageBox.warning(self, "提示", "缺少校准文件，请先进行麦克风校准。")
            return

        lumped_inputs = self._get_lumped_parameter_inputs()
        if lumped_inputs is None:
            self.test_state.setText("参数错误")
            return
        tube_temperature, s_sample_mm2, v_backing_cc = lumped_inputs
        f, Z_abs, Z_Re, Z_Im = utils.calculate_impedance_Lumped_parameter(
            mic2=self.mic2_data,
            mic1=self.mic1_data,
            mic2_cal=mic2_cal_data,
            mic1_cal=mic1_cal_data,
            sf=samplerate,
            temp=tube_temperature,
            s_sample_mm2=s_sample_mm2,
            v_backing_cc=v_backing_cc,
        )
        self.test_result = {
            "f": f,
            "Z_abs": Z_abs,
            "Z_Re": Z_Re,
            "Z_Im": Z_Im,
        }
        self.update_plot3_by_selector()

        self.logger.info(f"画图完成！")
        self.test_state.setText("测试完成")
        utils.set_run_button_enabled(self.run_test_button, True)
        AudioSessionManager.release(self)

    @staticmethod
    def generate_calibrated_signal(signal_info, samplerate):
        """
        生成校准后的刺激信号
        参数:
            signal_info: dict, 包含信号参数，其中 "signal_amplitude" 表示目标电压幅值 (V)
            samplerate: int, 采样率

        返回:
            success (bool): 是否成功
            data (ndarray): 已按校准幅值缩放后的信号数据
            cal_amplitude (float): 校准后的幅值（实际用于缩放的因子，单位 V）
            error_msg (str): 失败时的错误信息，否则为 None
        """
        params_voltage = signal_info["signal_amplitude"]
        scm = SoundcardCalibrationManager()
        # 用y=ax+b校准函数求幅值因子cal_amplitude
        calibrate_code, calibrate_result = scm.calibrate_amplitude(params_voltage)
        if calibrate_code != True:
            return False, None, None, f"校准幅值失败：{calibrate_result}"
        cal_amplitude, max_voltage = calibrate_result
        if params_voltage > max_voltage:
            return False, None, None, f"参数电压过大（{params_voltage}V），最大可达 {max_voltage}V。"
        # 信号归一化
        signal_info['signal_amplitude'] = 1
        _, data = utils.generate_chirp_wrapper(signal_info, samplerate)
        # 用幅值因子进行缩放
        data = data * cal_amplitude
        return True, data, cal_amplitude, None

    def update_time_plot(self, time, mic1_pa, mic2_pa):
        self.curve1.setData(time, mic1_pa)
        self.curve2.setData(time, mic2_pa)

    @staticmethod
    def load_calibration_data():
        # 读取校准数据
        cal_dir = os.path.join(os.getcwd(), "！校准")
        mic1_path = os.path.join(cal_dir, "MIC1.txt")
        mic2_path = os.path.join(cal_dir, "MIC2.txt")

        # 加载校准数据（float32 精度）
        mic1_cal = np.loadtxt(mic1_path, dtype=np.float32)
        mic2_cal = np.loadtxt(mic2_path, dtype=np.float32)
        return mic1_cal, mic2_cal

    @staticmethod
    def load_record_data():
        # 读取采集数据
        record_dir = os.path.join(os.getcwd(), "！采集")
        mic1_path = os.path.join(record_dir, "MIC1.txt")
        mic2_path = os.path.join(record_dir, "MIC2.txt")

        # 加载校准数据（float32 精度）
        mic1_record = np.loadtxt(mic1_path, dtype=np.float32)
        mic2_record = np.loadtxt(mic2_path, dtype=np.float32)
        return mic1_record, mic2_record

    def save_test_data_to_excel(self):
        if not self.test_result:
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return
        # 获取原始数据
        line_f = self.test_result["f"]

        # 线性==>对数
        def safe_abs_log(data):
            data = np.abs(np.asarray(data, dtype=float)).copy()
            data[data <= 0] = np.nan
            return np.log10(data)

        log_Z_abs = safe_abs_log(self.test_result["Z_abs"])
        log_Z_Re = safe_abs_log(self.test_result["Z_Re"])
        log_Z_Im = safe_abs_log(self.test_result["Z_Im"])

        # 应用Savitzky-Golay滤波器对对数数据进行平滑
        if self.soft_value > 3:
            log_Z_abs = savgol_filter(log_Z_abs, window_length=self.soft_value, polyorder=3)
            log_Z_Re = savgol_filter(log_Z_Re, window_length=self.soft_value, polyorder=3)
            log_Z_Im = savgol_filter(log_Z_Im, window_length=self.soft_value, polyorder=3)

        # 对数==>线性
        Z_abs = 10 ** log_Z_abs
        Z_Re = 10 ** log_Z_Re
        Z_Im = 10 ** log_Z_Im

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not save_dir:
            return  # 用户取消

        try:
            file1 = os.path.join(save_dir, "传输阻抗率测试结果.csv")
            with open(file1, "w", newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["f", "Z_abs", "Z_Re", "Z_Im"])
                for i in range(len(line_f)):
                    writer.writerow([line_f[i], Z_abs[i], Z_Re[i], Z_Im[i]])

            msg = f"保存成功！\n文件已保存至：\n{file1}"
            QMessageBox.information(self, "Success", msg)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")

    def update_plot3_by_selector(self, source=None):
        text = self.plot_type_selector.currentText()
        if "abs" in text:
            self.plot3.setLabel('left', 'Z abs', units='Rayl')
        elif "Re" in text:
            self.plot3.setLabel('left', 'Z Re', units='Rayl')
        elif "Im" in text:
            self.plot3.setLabel('left', 'Z Im', units='Rayl')
        else:
            QMessageBox.warning(self, "提示", f"无{text}选项")
            return

        if not self.test_result:
            if source == "config_tree":
                return
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return

        self._clear_history_lines()
        if self.crosshair_enabled:
            self.del_cross()

        try:
            self.plot3.scene().sigMouseClicked.disconnect(self.on_plot3_clicked)
        except Exception:
            pass
        self.plot3.scene().sigMouseClicked.connect(self.on_plot3_clicked)

        log_f, log_y = self._get_plot3_display_data(self.test_result)
        self.curve3.setData(log_f, log_y)
        self.curve3.show()

        self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)
        self.plot_state = True
        self.update_history_lines()

    def _legacy_update_plot3_by_selector(self):
        """
        根据平滑值来重新画图
        """
        text = self.plot_type_selector.currentText()
        if text == "传输阻抗率Z abs":
            self.plot3.setLabel('left', 'Z abs', units='Rayl')
        elif text == "传输阻抗率Z Re":
            self.plot3.setLabel('left', 'Z Re', units='Rayl')
        elif text == "传输阻抗率Z Im":
            self.plot3.setLabel('left', 'Z Im', units='Rayl')
        else:
            print("没这个选择项")
            return

        if not self.test_result:
            QMessageBox.warning(self, "提示", "无测试结果，请先进行测试！")
            return
        self.plot3.clear()

        # 十字线和提示框初始化但不添加到图上
        self.vLine = pyqtgraph.InfiniteLine(angle=90, movable=False, pen='r')
        self.hLine = pyqtgraph.InfiniteLine(angle=0, movable=False, pen='r')
        self.text = pyqtgraph.TextItem("", anchor=(0, 1), fill=pyqtgraph.mkBrush(255, 255, 255, 200), border='k')

        # 十字线显示开关
        self.crosshair_enabled = False

        # 绑定双击事件
        try:
            self.plot3.scene().sigMouseClicked.disconnect(self.on_plot3_clicked)
        except Exception:
            pass
        self.plot3.scene().sigMouseClicked.connect(self.on_plot3_clicked)

        if text == "传输阻抗率Z abs":
            log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
            log_Z_abs = np.asarray(self.test_result["Z_abs"], dtype=float).copy()
            log_Z_abs[log_Z_abs <= 0] = np.nan
            log_Z_abs = np.log10(log_Z_abs)
            if self.soft_value > 3:
                log_Z_abs = savgol_filter(log_Z_abs, window_length=self.soft_value, polyorder=3)
            self.plot3.plot(log_f, log_Z_abs, pen=mkPen(color='black', width=2))

        elif text == "传输阻抗率Z Re":
            log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
            log_Z_Re = np.abs(np.asarray(self.test_result["Z_Re"], dtype=float))
            log_Z_Re[log_Z_Re <= 0] = np.nan
            log_Z_Re = np.log10(log_Z_Re)
            if self.soft_value > 3:
                log_Z_Re = savgol_filter(log_Z_Re, window_length=self.soft_value, polyorder=3)
            self.plot3.plot(log_f, log_Z_Re, pen=mkPen(color='black', width=2))

        elif text == "传输阻抗率Z Im":
            log_f = np.log10(np.asarray(self.test_result["f"], dtype=float))
            log_Z_Im = np.abs(np.asarray(self.test_result["Z_Im"], dtype=float))
            log_Z_Im[log_Z_Im <= 0] = np.nan
            log_Z_Im = np.log10(log_Z_Im)
            if self.soft_value > 3:
                log_Z_Im = savgol_filter(log_Z_Im, window_length=self.soft_value, polyorder=3)
            self.plot3.plot(log_f, log_Z_Im, pen=mkPen(color='black', width=2))
        else:
            print("无")

        self.plot3.plotItem.enableAutoRange(axis='xy', enable=True)  # X 和 Y 轴的自动范围调整
        # self.plot3.plotItem.enableAutoRange(axis='y', enable=False)  # Y轴范围不变
        self.plot_state = True

    def del_cross(self):
        # 清竖线
        if hasattr(self, "vLine") and self.vLine is not None:
            self.plot3.removeItem(self.vLine)
            self.vLine = None

        # 清横线
        if hasattr(self, "hLine") and self.hLine is not None:
            self.plot3.removeItem(self.hLine)
            self.hLine = None

        # 清提示文字
        if hasattr(self, "text") and self.text is not None:
            self.plot3.removeItem(self.text)
            self.text = None

        # 清点击的小圆点
        if hasattr(self, "click_marker") and self.click_marker is not None:
            self.plot3.removeItem(self.click_marker)
            self.click_marker = None
        # 标记：当前没有十字线了
        self.crosshair_enabled = False

    def popup_pdf(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pdf_path = os.path.join(base_dir, "resources", "file", "小孔膜片阻抗管测试系统操作使用手册.pdf")
        if os.path.exists(pdf_path):
            if sys.platform.startswith("win"):
                os.startfile(pdf_path)  # Windows
            else:
                subprocess.run(["open", pdf_path])  # macOS 用 open；Linux 可用 xdg-open
        else:
            print("❌ PDF 文件不存在！")

