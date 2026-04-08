import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path


class CSVSplitterUtility:
    SAMPLING_RATE = 50  # Hz
    
    def __init__(self):
        self.csv_file = None
        self.num_rows = 0
        self.duration_seconds = 0
        
    def select_csv_file(self):
        """Let user select a CSV file"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if not file_path:
            return False
        
        self.csv_file = file_path
        return True
    
    def calculate_duration(self):
        """Calculate the duration of the CSV file"""
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                # Count rows (including header if present)
                rows = list(reader)
                
                # Check if first row is header (contains non-numeric values)
                try:
                    float(rows[0][0])
                    self.num_rows = len(rows)  # All rows are data
                except (ValueError, IndexError):
                    self.num_rows = len(rows) - 1  # First row is header
            
            self.duration_seconds = self.num_rows / self.SAMPLING_RATE
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file: {e}")
            return False
    
    def display_duration(self):
        """Display duration to user"""
        minutes = int(self.duration_seconds // 60)
        seconds = self.duration_seconds % 60
        print(f"\nFile: {os.path.basename(self.csv_file)}")
        print(f"Total data rows: {self.num_rows}")
        print(f"Duration: {minutes}m {seconds:.2f}s ({self.duration_seconds:.2f} seconds)")
    
    def get_time_period(self):
        """Get desired time period from user"""
        root = tk.Tk()
        root.withdraw()
        
        while True:
            try:
                time_seconds = simpledialog.askfloat(
                    "Time Period",
                    f"Enter desired time period in seconds (max {self.duration_seconds:.2f}s):",
                    minvalue=0.1
                )
                
                if time_seconds is None:  # User clicked Cancel
                    return None
                
                if time_seconds > self.duration_seconds:
                    messagebox.showwarning(
                        "Invalid Input",
                        f"Time period ({time_seconds}s) exceeds file duration ({self.duration_seconds:.2f}s)"
                    )
                    continue
                
                root.destroy()
                return time_seconds
                
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
                root.destroy()
                return None
    
    def split_csv(self, time_period_seconds):
        """Split CSV into pieces based on time period"""
        try:
            # Read all data
            with open(self.csv_file, 'r') as f:
                reader = csv.reader(f)
                all_rows = list(reader)
            
            # Separate header and data
            header = None
            data_rows = all_rows
            try:
                float(all_rows[0][0])
            except (ValueError, IndexError):
                header = all_rows[0]
                data_rows = all_rows[1:]
            
            # Calculate rows per piece
            rows_per_piece = int(time_period_seconds * self.SAMPLING_RATE)
            
            # Create output folder
            filename = Path(self.csv_file).stem
            output_folder = Path(self.csv_file).parent / f"{filename} pieces"
            output_folder.mkdir(exist_ok=True)
            
            # Split and save pieces
            piece_count = 0
            for i in range(0, len(data_rows), rows_per_piece):
                piece_data = data_rows[i:i + rows_per_piece]
                piece_count += 1
                
                piece_filename = output_folder / f"{filename}_piece_{piece_count}.csv"
                
                with open(piece_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if header:
                        writer.writerow(header)
                    writer.writerows(piece_data)
                
                piece_duration = len(piece_data) / self.SAMPLING_RATE
                print(f"  Piece {piece_count}: {len(piece_data)} rows ({piece_duration:.2f}s)")
            
            print(f"\n✓ Successfully created {piece_count} pieces in:")
            print(f"  {output_folder}")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not split CSV: {e}")
            return False
    
    def run(self):
        """Main execution flow"""
        print("=" * 60)
        print("CSV SPLITTER UTILITY (50Hz Accelerometer Data)")
        print("=" * 60)
        
        # Select file
        if not self.select_csv_file():
            print("No file selected. Exiting.")
            return
        
        # Calculate duration
        if not self.calculate_duration():
            print("Could not analyze file. Exiting.")
            return
        
        self.display_duration()
        
        # Get time period from user
        time_period = self.get_time_period()
        if time_period is None:
            print("Cancelled by user.")
            return
        
        print(f"\nSplitting into pieces of {time_period}s...")
        
        # Split the CSV
        if self.split_csv(time_period):
            messagebox.showinfo(
                "Success",
                f"CSV split successfully!\nCreated pieces in the output folder."
            )
        else:
            print("Failed to split CSV.")


if __name__ == "__main__":
    utility = CSVSplitterUtility()
    utility.run()
