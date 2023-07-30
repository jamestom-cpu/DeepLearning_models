import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout

def create_app():
    app = QApplication(sys.argv)
    return app

def main():
    app = create_app()

    # Create the main window
    window = QWidget()
    window.setWindowTitle("Different-sized Widgets")
    window.setGeometry(100, 100, 400, 300)  # Set the initial window size

    # Create widgets
    label1 = QLabel("Label 1 (Default Size)")
    label2 = QLabel("Label 2 (Bigger Font)")
    label2.setStyleSheet("font-size: 16px;")  # Increase the font size of label2

    button1 = QPushButton("Button 1 (Default Size)")
    button2 = QPushButton("Button 2 (Bigger Size)")
    button2.setStyleSheet("font-size: 16px;")  # Increase the font size of button2

    input1 = QLineEdit()
    input1.setPlaceholderText("Input 1 (Default Size)")
    input2 = QLineEdit()
    input2.setPlaceholderText("Input 2 (Bigger Size)")
    input2.setStyleSheet("font-size: 16px;")  # Increase the font size of input2

    # Create a layout to arrange the widgets vertically
    layout = QVBoxLayout()
    layout.addWidget(label1)
    layout.addWidget(label2)
    layout.addWidget(button1)
    layout.addWidget(button2)
    layout.addWidget(input1)
    layout.addWidget(input2)

    # Set the layout for the main window
    window.setLayout(layout)

    # Show the window
    window.show()

    # Start the application's event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
