import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import defaultdict

class LogVisualizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Log File Visualizer")
        master.geometry("1000x700")

        self.df = None
        self.log_file_path = ""
        self.plottable_columns = []

        # --- Top Frame for File Loading ---
        top_frame = ttk.Frame(master, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Button(top_frame, text="Load Log File", command=self.load_file).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(top_frame, text="No file loaded.")
        self.file_label.pack(side=tk.LEFT, padx=5)

        # --- Main PanedWindow for Controls and Plot ---
        main_paned_window = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Frame for Controls (Column Selection) ---
        controls_frame = ttk.Frame(main_paned_window, width=250, height=600, relief=tk.RIDGE)
        main_paned_window.add(controls_frame, weight=1)

        ttk.Label(controls_frame, text="Select Columns to Plot:", font=('Arial', 12)).pack(pady=10)

        self.listbox_frame = ttk.Frame(controls_frame)
        self.listbox_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.columns_listbox = tk.Listbox(self.listbox_frame, selectmode=tk.MULTIPLE, exportselection=False)
        self.columns_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.listbox_frame, orient=tk.VERTICAL, command=self.columns_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.columns_listbox.config(yscrollcommand=scrollbar.set)
        
        self.plot_button = ttk.Button(controls_frame, text="Plot Selected", command=self.plot_selected_data, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        # --- Right Frame for Plot ---
        plot_frame_container = ttk.Frame(main_paned_window, relief=tk.RIDGE)
        main_paned_window.add(plot_frame_container, weight=4)
        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame_container)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def parse_log_data(self, file_path):
        data_list = []
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        log_entry = json.loads(line)
                        
                        # Expand test_coco_eval_bbox if it exists and is a list of 12
                        if 'test_coco_eval_bbox' in log_entry and \
                           isinstance(log_entry['test_coco_eval_bbox'], list) and \
                           len(log_entry['test_coco_eval_bbox']) == 12:
                            for j, val in enumerate(log_entry['test_coco_eval_bbox']):
                                log_entry[f'test_coco_eval_bbox_{j}'] = val
                            # del log_entry['test_coco_eval_bbox'] # Optional: remove original
                        
                        data_list.append(log_entry)
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON on line {i+1}")
            
            if not data_list:
                return None

            df = pd.DataFrame(data_list)
            
            # Ensure 'epoch' is numeric and sort by it
            if 'epoch' in df.columns:
                df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
                df = df.sort_values(by='epoch').dropna(subset=['epoch'])
            else:
                print("Warning: 'epoch' column not found in log data.")
                return None
                
            return df
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {file_path}")
            return None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse log file: {e}")
            return None

    def load_file(self):
        filepath = filedialog.askopenfilename(
            title="Open Log File",
            filetypes=(("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.log_file_path = filepath
        self.df = self.parse_log_data(self.log_file_path)

        if self.df is not None:
            self.file_label.config(text=f"Loaded: {self.log_file_path.split('/')[-1]}")
            self.populate_column_list()
            self.plot_button.config(state=tk.NORMAL)
            self.ax.clear() # Clear previous plot
            self.canvas.draw()
        else:
            self.file_label.config(text="Failed to load file.")
            self.columns_listbox.delete(0, tk.END)
            self.plot_button.config(state=tk.DISABLED)


    def populate_column_list(self):
        self.columns_listbox.delete(0, tk.END)
        self.plottable_columns = []
        if self.df is not None:
            for col in self.df.columns:
                # Consider numeric columns, excluding 'epoch' and 'n_parameters'
                if pd.api.types.is_numeric_dtype(self.df[col]) and col not in ['epoch', 'n_parameters']:
                    # Check if column has at least one non-NaN value
                    if self.df[col].notna().any():
                        self.plottable_columns.append(col)
            
            # Sort columns for better readability, e.g., train metrics together
            self.plottable_columns.sort() 
            for col_name in self.plottable_columns:
                self.columns_listbox.insert(tk.END, col_name)

    def plot_selected_data(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a log file first.")
            return

        selected_indices = self.columns_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one column to plot.")
            return

        selected_cols = [self.columns_listbox.get(i) for i in selected_indices]

        self.ax.clear()
        
        for col in selected_cols:
            if col in self.df.columns and 'epoch' in self.df.columns:
                # Drop NaN values for plotting to avoid gaps or errors for that specific line
                plot_data = self.df[['epoch', col]].dropna()
                self.ax.plot(plot_data['epoch'], plot_data[col], label=col, marker='.')
        
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Log Metrics Over Epochs")
        self.ax.legend(loc='best')
        self.ax.grid(True)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LogVisualizerApp(root)
    root.mainloop()
