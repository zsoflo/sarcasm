import pandas as pd
import os
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout



# pip install pandas
# pip install PyQt5
# replace file name by your own

file_path = "C:\\Users\\hfall\\Documents\\M1_TAL_IDMC\\Cours\\S7\\Projet\\annotation\\samba.csv"
save_path = file_path.split('.csv')[0] + '_save.csv'
out_path = file_path.split('.csv')[0] + '_out.csv'

if os.path.isfile(out_path) :
	df = pd.read_csv(out_path)
else :
	df = pd.read_csv(file_path)

size = len(df)

dict_df = df.to_dict()

def save() :
	pd.DataFrame(dict_df).to_csv(save_path, index=False)

class SaveWindow(QWidget) :
	def closeEvent(self, event) :
		pd.DataFrame(dict_df).to_csv(out_path, index=False)
		print("Fichier sauvegardé")
		super().closeEvent(event)


app = QApplication([])
mainwindow = SaveWindow()
mainwindow.setWindowTitle(f"Annotation : {file_path}")
mainwindow.setFixedSize(1020,800)
layout = QVBoxLayout()
numlabel = QLabel()
layout.addWidget(numlabel)
textlabel = QLabel(wordWrap=True)
layout.addWidget(textlabel)
sublayout = QHBoxLayout()
yes_button = QPushButton("1")
sublayout.addWidget(yes_button)
no_button = QPushButton("0")
sublayout.addWidget(no_button)
layout.addLayout(sublayout)
	
def annotate(annot, index) :
	dict_df['annotation'][index] = 1 if annot else 0
	save()
	load_row(index + 1)

def load_row(index) :
	row = df.loc[index]
	if index >= size :
		window.close()
	elif str(row['annotation']) in ['0','1','1.0', '0.0'] :
		load_row(index + 1)
	else :
		yes_button.disconnect()
		no_button.disconnect()
		yes_button.clicked.connect( lambda : annotate(True, index)  )
		no_button.clicked.connect( lambda : annotate(False, index)  )
		numlabel.setText(f"Tweet {int(row['ID'])} : {index + 1}/{size}")
		textlabel.setText(row['tweet'])


mainwindow.setLayout(layout)
mainwindow.show()
load_row(0)
app.exec()