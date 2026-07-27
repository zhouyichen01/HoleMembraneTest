import copy

from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.uic import loadUi

from control.log_manager import LogManager
from control.utils import utils
from custom.customSignals import sign


class ConfigTreeInterface(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.ui = None
        self.logger = LogManager.set_log_handler("配置树")
        self.init_ui()
        self.init_fun()

    def init_ui(self):
        ui_file = QFile(":ui/tree_config.ui")
        if not ui_file.exists():
            raise FileNotFoundError("未找到资源文件 tree_config.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = loadUi(ui_file, self)
        ui_file.close()
        self.setWindowTitle("配置树")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.resize(620, 520)

    def init_fun(self):
        self.treeWidget.itemClicked.connect(self.on_tree_click)
        self.buttonGroup.addButton(self.radioButton, 1)
        self.buttonGroup.addButton(self.radioButton_2, 0)
        self.buttonGroup.buttonClicked[int].connect(self.on_group_clicked)
        self.stackedWidget.setCurrentIndex(0)
        self.fill_page()

    @staticmethod
    def _default_config():
        return {
            "line": {
                "is_display_history_line": 1,
                "manage_history_line": {},
                "saved_history_line": [],
            }
        }

    def _read_config(self):
        res, content = utils.get_config_content("tree_config.json")
        if not res or not isinstance(content, dict):
            content = self._default_config()
        line_cfg = content.setdefault("line", {})
        line_cfg.setdefault("is_display_history_line", 1)
        line_cfg.setdefault("manage_history_line", {})
        line_cfg.setdefault("saved_history_line", [])
        return content

    def _write_config(self, content):
        return utils.write_config_content("tree_config.json", content)

    def on_tree_click(self, item, column=None):
        text = item.text(0)
        if text == "显示开关":
            self.ui.stackedWidget.setCurrentIndex(0)
            self.fill_page()
        elif text == "历史线":
            self.ui.stackedWidget.setCurrentIndex(1)
            self.fill_page_2()

    def fill_page(self):
        content = self._read_config()
        val = content.get("line", {}).get("is_display_history_line", 1)
        if val:
            self.radioButton.setChecked(True)
        else:
            self.radioButton_2.setChecked(True)

    def on_group_clicked(self, bid):
        content = self._read_config()
        content["line"]["is_display_history_line"] = int(bid)
        if not self._write_config(content):
            QMessageBox.warning(self, "错误", "tree_config.json 写入失败")
            return
        sign.update_plot3_by_selector_sign.emit("config_tree")

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()

    def _clear_group_box(self):
        old_layout = self.ui.groupBox.layout()
        if old_layout is not None:
            self._clear_layout(old_layout)
            QWidget().setLayout(old_layout)

    def fill_page_2(self):
        content = self._read_config()
        manage = content.get("line", {}).get("manage_history_line", {})

        self._clear_group_box()
        layout = QVBoxLayout(self.ui.groupBox)
        layout.setContentsMargins(0, 0, 0, 0)

        if not manage:
            tip = QLabel("暂无历史线")
            tip.setAlignment(Qt.AlignCenter)
            tip.setStyleSheet("color: #9AA0A6; font-size: 9pt; padding: 8px;")
            layout.addStretch()
            layout.addWidget(tip)
            layout.addStretch()
            self.ui.groupBox.setLayout(layout)
            return

        for key, item in manage.items():
            state = str(item.get("state", "False")) == "True"
            colour = item.get("colour", "#1f77b4")
            line_name = item.get("line_name") or f"历史线{key}"

            row = QHBoxLayout()

            cb = QCheckBox()
            cb.setChecked(state)
            cb.stateChanged.connect(lambda state, index=key: self.on_history_line_toggle(index, state))

            label_name = QLabel(f"{line_name}: ")
            label_name.setFixedWidth(140)
            label_name.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            color_line = QFrame()
            color_line.setFixedSize(24, 20)
            color_line.setFrameShape(QFrame.HLine)
            color_line.setStyleSheet(f"background-color: {colour}; border: none;")

            btn_rename = QToolButton()
            btn_rename.setText("重命名")
            btn_rename.setFixedWidth(64)
            btn_rename.clicked.connect(
                lambda _, index=key, name_object=label_name: self.on_history_line_rename(index, name_object)
            )

            btn_delete = QToolButton()
            btn_delete.setText("删除")
            btn_delete.setFixedWidth(52)
            btn_delete.clicked.connect(lambda _, index=key, name=line_name: self.delete_history_line(index, name))

            row.addWidget(cb)
            row.addWidget(label_name)
            row.addWidget(color_line)
            row.addWidget(btn_rename)
            row.addWidget(btn_delete)
            row.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
            layout.addLayout(row)

        layout.addStretch()
        self.ui.groupBox.setLayout(layout)

    def on_history_line_rename(self, index, name_object):
        content = self._read_config()
        current = name_object.text().rstrip(": ")
        new_name, ok = QInputDialog.getText(self, "重命名", "名称：", text=current)
        if not ok:
            return
        new_name = (new_name or "").strip()
        if not new_name:
            QMessageBox.warning(self, "格式错误", "名称不能为空。")
            return
        if len(new_name) > 24:
            QMessageBox.warning(self, "格式错误", "名称长度不能超过 24 个字符。")
            return

        try:
            content["line"]["manage_history_line"][str(index)]["line_name"] = new_name
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"更新名称失败：{exc}")
            return

        if self._write_config(content):
            name_object.setText(f"{new_name}: ")
            sign.update_plot3_by_selector_sign.emit("config_tree")
        else:
            QMessageBox.warning(self, "错误", "保存名称失败。")

    def on_history_line_toggle(self, index, state):
        content = self._read_config()
        try:
            content["line"]["manage_history_line"][str(index)]["state"] = (
                "True" if state == Qt.Checked else "False"
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"更新历史线状态失败：{exc}")
            return

        if self._write_config(content):
            sign.update_plot3_by_selector_sign.emit("config_tree")
        else:
            QMessageBox.warning(self, "错误", "保存历史线状态失败。")

    def delete_history_line(self, key, name):
        user_choice = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除“{name}”吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if user_choice != QMessageBox.Yes:
            return

        content = self._read_config()
        backup_content = copy.deepcopy(content)
        try:
            del content["line"]["manage_history_line"][str(key)]
            del content["line"]["saved_history_line"][int(key)]
            old_lines = content["line"]["manage_history_line"]
            content["line"]["manage_history_line"] = {
                str(new_index): old_lines[old_key]
                for new_index, old_key in enumerate(sorted(old_lines, key=lambda item: int(item)))
            }
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"删除失败：{exc}")
            return

        if self._write_config(content):
            self.fill_page_2()
            sign.update_plot3_by_selector_sign.emit("config_tree")
            return

        self._write_config(backup_content)
        QMessageBox.warning(self, "错误", "删除失败，已尝试恢复删除前配置。")
