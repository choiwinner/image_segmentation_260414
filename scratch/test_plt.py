import os
os.environ['QT_API'] = 'pyside6'
import PySide6
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

print("Matplotlib backend:", matplotlib.get_backend())
try:
    fig, ax = plt.subplots()
    print("Successfully created subplots")
    plt.close(fig)
    print("Successfully closed figure")
except Exception as e:
    print(f"Error: {e}")
