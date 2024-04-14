import GUI as gui 
import sys
from PyQt6.QtWidgets import QApplication

# Create the application object
app = QApplication(sys.argv)

# Create a window object
window = gui.MyWindow()

# Show the window
window.show()

# Run the event loop
sys.exit(app.exec())
