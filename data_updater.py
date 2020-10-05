from fire import Fire
import schedule
import time

try:
    f = Fire()
    hour = d.current_hour()
    f.generate_schools_in_danger()
    f.save_schools()
    f.generate_new_image()
    
except Exception as e: 
    print(f"Error: {e}")
    time.sleep(60)

while hour != f.current_hour():
    try:
        f = Fire()
        hour = d.current_hour()
        f.generate_schools_in_danger()
        f.save_schools()
        f.generate_new_image()
        hour = f.current_hour()
    except Exception as e: 
        print(f"Error: {e}")
        time.sleep(3600)
        