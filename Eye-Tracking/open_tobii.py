import os

def open_app(route_direct_access):
    """Try open application with the given shortcut path."""
    try:
        os.startfile(route_direct_access)
        print(f"Opening Tobii Pro Spark Manager...")
    except OSError as e:
        print(f"Error: opening Tobii Pro Spark Manager - {e}")

if __name__ == "__main__":
    #route where the tobii pro eye tracker manager is located 
    route_lnk = r"c:\Users\marta\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Tobii Pro Eye Tracker Manager.lnk"
    open_app(route_lnk)