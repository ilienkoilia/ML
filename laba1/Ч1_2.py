from logging import root
import requests
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.dates
from datetime import datetime

def draw(name, valute):

    date = datetime.now()
    date_year = date.year
    date_month = date.month
    date_day = date.day
    result = requests.get(f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/{date_year}&date_req2={date_day}/{date_month}/{date_year}&VAL_NM_RQ={valute}")

    root = ET.fromstring(result.text)

    value=[[],[]]
    

    np.savetxt(f'{name}.csv', (value[0],value[1]), delimiter=',')
    
    xlist=[]
    ylist=[]
    
    for date_ in root:
        value[0].append(date_.get('Date'))
        value[1].append(date_[1].text) 

    for val0 in value[0]:
        date=datetime.strptime(val0, '%d.%m.%Y')
        xlist.append(matplotlib.dates.date2num(date))

    for val1 in value[1]:
        ylist.append(float(val1.replace(',','.')))

    fig, ax = plt.subplots()

    plt.title('Курс '+name+' в ' + str(date_year))
    plt.ylabel('Цена за '+name)
    plt.xlabel('Месяц')

    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(1))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m'))

    ax.plot(xlist, ylist)

    plt.show()
    

draw("USD", "R01235")
draw("JPY", "R01820")
draw("EUR", "R01239")
draw("UAH", "R01720")