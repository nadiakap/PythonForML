import pandas as pd

REGULAR_HOURS = 40

def compute_salary(x):
   
    if x['hours']>REGULAR_HOURS:
        return REGULAR_HOURS*x['rate'] + (x['hours']-REGULAR_HOURS)*x['overtime_rate']
    else:
        return x['rate']*x['hours']
      
    
salary_df = pd.read_excel('WeeklySalary.xlsx')
salary_df['payment']=salary_df.apply(lambda x:compute_salary(x),axis=1)
print(salary_df)
