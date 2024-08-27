from datetime import datetime

def convert_date_format(date_str):
    try:
        # Convertir la date du format DD/MM/YYYY à YYYY-MM-DD
        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Format de date invalide : {date_str}")
