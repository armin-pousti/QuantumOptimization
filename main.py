import portfolio

def run(input_data,solver_params,extra_arguments):
  # Check if evaluation_date is in extra_arguments first
  if 'evaluation_date' in extra_arguments:
    input_data['evaluation_date']=extra_arguments['evaluation_date']
  # If no evaluation_date, check if there's a "from" field in input_data
  elif 'evaluation_date' not in input_data and 'from' in input_data:
    input_data['evaluation_date'] = input_data['from']
  # If still no evaluation_date, use today's date as fallback
  elif 'evaluation_date' not in input_data:
    import datetime
    input_data['evaluation_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
  
  return portfolio.run(input_data)
