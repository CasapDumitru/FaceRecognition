import xlrd
import xlsxwriter

# read the model feature from excel and store it in a matrix
def readMatFromExcell(fileName):
	wkb=xlrd.open_workbook(fileName)
	sheet=wkb.sheet_by_index(0)

	matrix=[]
	for row in range (sheet.nrows):
		_row = []
		for col in range (sheet.ncols):
			_row.append(sheet.cell_value(row,col))
		matrix.append(_row)

	return matrix;


# write the model feature from matrix in excel
def writeMatToExcell(fileName,matrix):
	workbook = xlsxwriter.Workbook(fileName)
	worksheet = workbook.add_worksheet()

	col = 0

	for row, data in enumerate(matrix):
		worksheet.write_row(row, col, data)

	workbook.close()