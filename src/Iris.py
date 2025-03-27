import sys
import os
import random
from PyQt6.QtWidgets import QApplication, QLabel, QMenu, QDialog, QVBoxLayout, QTextEdit, QPushButton, QWidget
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt6.QtGui import QMovie, QIcon
from chat import Iris
from mutiModalIngest import main_loop as data_loop
import threading

# Animation folders (same as original)
animationsFolder = os.path.join('..', 'animations')
bye_Animations_Folder = os.path.join(animationsFolder, 'bye')
cry_Animations_Folder = os.path.join(animationsFolder, 'cry')
dance_Animations_Folder = os.path.join(animationsFolder, 'dance')
fight_Animations_Folder = os.path.join(animationsFolder, 'fight')
fishy_Animations_Folder = os.path.join(animationsFolder, 'fishy')
goofy_Animations_Folder = os.path.join(animationsFolder, 'goofy')
hover_Animations_Folder = os.path.join(animationsFolder, 'hover')
idle_Animations_Folder = os.path.join(animationsFolder, 'idle')
party_Animations_Folder = os.path.join(animationsFolder, 'party')
sing_Animations_Folder = os.path.join(animationsFolder, 'sing')
thinking_Animations_Folder = os.path.join(animationsFolder, 'thinking')
walk_Animations_Folder = os.path.join(animationsFolder, 'walk')

animation_folders = {
    "bye": bye_Animations_Folder,
    "cry": cry_Animations_Folder,
    "dance": dance_Animations_Folder,
    "fight": fight_Animations_Folder,
    "fishy": fishy_Animations_Folder,
    "goofy": goofy_Animations_Folder,
    "hover": hover_Animations_Folder,
    "idle": idle_Animations_Folder,
    "party": party_Animations_Folder,
    "sing": sing_Animations_Folder,
    "think": thinking_Animations_Folder,
    "walk": walk_Animations_Folder
}

stop_event = threading.Event()

class QueryThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        try:
            result = Iris(self.query)
            if isinstance(result, dict):
                self.result_signal.emit(result.get('answer', 'No answer'))
            else:
                self.result_signal.emit(str(result))
        except Exception as e:
            self.result_signal.emit(f"Error: {str(e)}")

class QueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Query")
        self.setFixedSize(400, 600)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main layout
        layout = QVBoxLayout(self)

        # Main widget with rounded corners and gradient
        main_widget = QWidget(self)
        main_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e1e1e, stop:1 #2d2d2d);
                border-radius: 20px;
            }
        """)
        layout.addWidget(main_widget)

        # Layout for content
        content_layout = QVBoxLayout(main_widget)

        # Header
        header = QLabel("Query Assistant")
        header.setStyleSheet("color: #ffffff; font-size: 16pt; font-weight: bold; background: transparent; padding: 10px;")
        content_layout.addWidget(header)

        # Query input
        self.query_text = QTextEdit()
        self.query_text.setFixedHeight(100)
        self.query_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 5px;
                font-size: 12pt;
            }
        """)
        content_layout.addWidget(self.query_text)

        # Loading label
        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #bbbbbb; font-size: 11pt; background: transparent; padding: 5px;")
        content_layout.addWidget(self.loading_label)

        # Result area
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #444444;
                border-radius: 5px;
                padding: 5px;
                font-size: 12pt;
            }
        """)
        content_layout.addWidget(self.result_text)

        # Buttons
        button_layout = QVBoxLayout()
        self.submit_button = QPushButton("Submit")
        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #007bff;
                color: #ffffff;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
        """)
        self.submit_button.clicked.connect(self.submit_query)
        button_layout.addWidget(self.submit_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #ffffff;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3c3c3c;
            }
        """)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        content_layout.addLayout(button_layout)

        # Center the dialog
        self.move(QApplication.primaryScreen().geometry().center() - self.rect().center())

    def submit_query(self):
        query = self.query_text.toPlainText().strip()
        if not query:
            self.result_text.setText("Enter a query.")
            return
        if len(query) > 1000:
            self.result_text.setText("Query too long (max 1000 chars).")
            return

        self.submit_button.setEnabled(False)
        self.loading_label.setText("Processing...")
        self.result_text.clear()

        self.query_thread = QueryThread(query)
        self.query_thread.result_signal.connect(self.update_result)
        self.query_thread.start()

    def update_result(self, result):
        self.loading_label.setText("")
        self.submit_button.setEnabled(True)
        self.result_text.setText(result)

class AnimatedBuddy(QLabel):
    def __init__(self, data_thread):
        super().__init__()
        self.setFixedSize(128, 128)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.data_thread = data_thread

        # Load animations
        self.animations = {}
        self.idle_animations = []
        self.current_idle_index = 0
        self.load_all_animations()

        # Set up the movie for animations
        self.movie = QMovie()
        self.movie.setScaledSize(self.size())
        # self.movie.frameChanged.connect(self.on_frame_changed)
        self.setMovie(self.movie)

        # Connect error signal to debug issues
        self.movie.error.connect(self.on_movie_error)

        # Start with idle animation
        self.current_animation = "idle"
        self.change_animation("idle")

        # Dragging functionality
        self.old_pos = None
        self.setMouseTracking(True)

        # Context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # Animation cycle timer (every 30 seconds)
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.change_animation_random)
        self.animation_timer.start(30000)

        # Move to initial position
        self.move(500, 300)

    def on_movie_error(self, error):
        print(f"QMovie Error: {error}")

    def on_frame_changed(self, frame_number):
        print(f"Frame changed: {frame_number}/{self.movie.frameCount()}")

    def load_all_animations(self):
        """Loads one random GIF for most animations, but all GIFs for idle."""
        for name, folder_path in animation_folders.items():
            if not os.path.exists(folder_path):
                print(f"Error: Folder '{folder_path}' not found!")
                self.animations[name] = []
                continue

            gif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.gif')]
            if not gif_files:
                print(f"No GIF files found in '{folder_path}'!")
                self.animations[name] = []
                continue

            if name == "idle":
                for gif_file in gif_files:
                    gif_path = os.path.join(folder_path, gif_file)
                    # Verify the file can be loaded by QMovie
                    movie = QMovie(gif_path)
                    if movie.isValid():
                        self.idle_animations.append(gif_path)
                        print(f"Loaded idle animation: {gif_file}")
                    else:
                        print(f"Failed to load idle animation: {gif_file}")
                    movie.deleteLater()
                self.animations[name] = self.idle_animations
            else:
                random_gif = random.choice(gif_files)
                gif_path = os.path.join(folder_path, random_gif)
                movie = QMovie(gif_path)
                if movie.isValid():
                    self.animations[name] = gif_path
                    print(f"Loaded {name} animation: {random_gif}")
                else:
                    print(f"Failed to load {name} animation: {random_gif}")
                movie.deleteLater()

    def change_animation(self, animation_name):
        self.movie.stop()
        if animation_name == "idle" and self.idle_animations:
            self.current_idle_index = random.randrange(len(self.idle_animations))
            self.current_animation = animation_name
            gif_path = self.idle_animations[self.current_idle_index]
            self.movie.setFileName(gif_path)
            if self.movie.isValid():
                self.movie.start()
                print(f"Switched to idle animation #{self.current_idle_index}: {gif_path}")
            else:
                print(f"Failed to play idle animation #{self.current_idle_index}: {gif_path}")
        elif animation_name in self.animations and self.animations[animation_name]:
            self.current_animation = animation_name
            gif_path = self.animations[animation_name]
            self.movie.setFileName(gif_path)
            if self.movie.isValid():
                self.movie.start()
                print(f"Switched to {animation_name} animation: {gif_path}")
                # Use finished signal instead of a timer
                self.movie.finished.connect(self.return_to_random_idle)
            else:
                print(f"Failed to play {animation_name} animation: {gif_path}")

    def change_animation_random(self):
        """Switches to a random animation (excluding idle and walk) and returns to a random idle."""
        valid_animations = {k: v for k, v in self.animations.items() if v and k not in ["idle", "walk", "bye"]}
        if valid_animations:
            animation_name = random.choice(list(valid_animations.keys()))
            self.change_animation(animation_name)

    def return_to_random_idle(self):
        """Returns to a random idle animation after a random animation completes."""
        if self.idle_animations:
            new_idle_index = random.randrange(len(self.idle_animations))
            while new_idle_index == self.current_idle_index and len(self.idle_animations) > 1:
                new_idle_index = random.randrange(len(self.idle_animations))
            self.current_idle_index = new_idle_index
            self.current_animation = "idle"
            gif_path = self.idle_animations[self.current_idle_index]
            self.movie.setFileName(gif_path)
            if self.movie.isValid():
                self.movie.start()
                print(f"Returned to idle animation #{self.current_idle_index}: {gif_path}")
            else:
                print(f"Failed to return to idle animation #{self.current_idle_index}: {gif_path}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event):
        self.old_pos = None

    def show_context_menu(self, pos):
        """Displays the context menu on right-click."""
        menu = QMenu(self)
        animations_menu = menu.addMenu("Animations")
        animations_menu.addAction("Random", self.change_animation_random)
        for anim_name in animation_folders.keys():
            animations_menu.addAction(anim_name.capitalize(), lambda name=anim_name: self.change_animation(name))
        menu.addAction("Query", self.show_query_dialog)
        menu.addAction("Quit", self.quit_with_bye)
        menu.exec(self.mapToGlobal(pos))

    def show_query_dialog(self):
        """Opens the query dialog."""
        dialog = QueryDialog(self)
        dialog.exec()

    def quit_with_bye(self):
        """Plays the 'bye' animation, stops the data thread, and quits."""
        if "bye" in self.animations and self.animations["bye"]:
            self.change_animation("bye")
            duration = self.movie.frameCount() * 100  # Approximate duration in milliseconds
            # Signal the thread to stop
            stop_event.set()
            # Wait for the thread to finish (up to 2 seconds) before quitting
            QTimer.singleShot(duration, lambda: self.finalize_quit())
        else:
            print("No 'bye' animation available, stopping thread and quitting immediately.")
            stop_event.set()
            self.finalize_quit()

    def finalize_quit(self):
        """Waits for the thread to finish and then quits the application."""
        if self.data_thread.is_alive():
            self.data_thread.join(timeout=2)  # Wait up to 2 seconds
            if self.data_thread.is_alive():
                print("Warning: Data thread did not terminate in time.")
            else:
                print("Data thread terminated successfully.")
        QApplication.quit()

if __name__ == "__main__":
    # Start the data ingestion thread with the stop event
    dataThread = threading.Thread(target = data_loop, args=(stop_event,))
    dataThread.start()

    # Launch the Animated Buddy
    app = QApplication(sys.argv)
    buddy = AnimatedBuddy(dataThread)
    buddy.show()
    sys.exit(app.exec())