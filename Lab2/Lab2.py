import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('Reading files')
df = pd.read_csv('sales_data.csv')
print(df)
total_profit = df['total_profit']
months = df['month_number']
profit_range = [150e3,175e3,200e3,225e3,250e3,300e3,350e3]
plt.figure()
plt.hist(total_profit,profit_range,label='Profit data')
plt.xlabel('profit range / $')
plt.ylabel('Actual profit / $')
plt.legend(loc='upper left')
plt.title('Profit data')

print('Exercise 7')
months = df['month_number']
bathing_soap = df['bathingsoap']
face_wash = df['facewash']
f, axs = plt.subplots(2,1,sharex=True)
axs[0].plot(months,bathing_soap,label='Bathing soap',color='k',marker='o',linewidth=3)
axs[0].set_title('Sales data of bathing soap')
axs[0].grid(True,linewidth=0.5,linestyle='--')
axs[0].legend()
axs[1].plot(months,face_wash,label='Facewash',color='r',marker='o',linewidth=3)
axs[1].set_title('Sales data of face soap')
axs[1].grid(True,linewidth=0.5,linestyle='--')
axs[1].legend()
plt.xticks(months)
plt.xlabel('Month number')
plt.ylabel('Sales unit in number')

#plt.show()


print('Exercise set 5')
print('Exercise 1')

import json
json_obj = '{"Name":"David", "Class":"I", "Age":6 }'

python_obj = json.loads(json_obj)
print('\nJSON data:')
print('\nName:', python_obj['Name'])
print('\nClass:', python_obj['Class'])
print('\nAge:', python_obj['Age'])

print('Exercise 2')
python_obj = { 'name': 'David', 'class': 'I', 'age': 6 }
print(python_obj)
json_obj = json.dumps(python_obj)
print(json_obj)
print(type(json_obj))

print('Exercise 3')
py_dict = { 'name': 'David', 'class': 'I', 'age': 6 }
py_list = ['Red',4,'ciao']
py_str = 'Python Json'
py_int = 4
py_float = 4.65
py_t = True
py_n = None
json_dict = json.dumps(py_dict)
json_list = json.dumps(py_list)
json_str = json.dumps(py_str)
json_int = json.dumps(py_int)
json_float = json.dumps(py_float)
json_t = json.dumps(py_t)
json_n = json.dumps(py_n)
print('json dict: ', json_dict)
print('json list: ', json_list)
print('json string: ', json_str)
print('json int: ', json_int)
print('json float: ', json_float)
print('json true: ', json_t)
print('json none: ', json_n)

print('Exercise 4')
py_dict = {'4': 2, '6': 7, '1':3, '2':4}
print('Original Dict: ', py_dict)
print('\nJson data:')
print(json.dumps(py_dict,sort_keys=True,indent=4))

print('\nExercise 5')
f = open('states.json')
states_data = json.load(f)
f.close()
print(states_data)
print(states_data['states'])
print('Json keys: ', [state.keys() for state in states_data['states']][0])
for state in states_data['states']:
    del state['area_codes']
print('New Json keys: ', [state.keys() for state in states_data['states']][0])
f = open('new_states.json','w')
json.dump(states_data,f,indent=2)
f.close()
f=open('new_states.json')
states_data = json.load(f)
f.close()
print('\nReloaded Json:', states_data['states'])
print('Json keys:', [state.keys() for state in states_data['states']][0])


