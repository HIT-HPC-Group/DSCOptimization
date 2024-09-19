import openpyxl

wb = openpyxl.load_workbook('choice.xlsx')
sheet = wb.worksheets[0]
col = ['E','H','K','N','Q']

with open('./parameters.txt', 'w') as w:
    for c in col:
        for i in range(3, 48):        
            s = ''
            s += str(sheet['B'+str(i)].value)
            s += ' '
            s += str(sheet['C'+str(i)].value)
            s += ' '
            s += str(sheet['D'+str(i)].value)
            s += ' '
            s += str(sheet[c + '1'].value[sheet[c + '1'].value.find('= ') + 2:])
            tmp = str(sheet[c + str(i)].value)
            tmp = tmp.replace(' ', '').split(',')
            for j in tmp:
                s = s + ' ' + str(j)
            s += '\n'
            w.write(s)
