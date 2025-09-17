import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import pandas as pd
import json
import threading
import time

class PerceptronSimple:
    def __init__(self, n_inputs):
        """Inicializa el perceptrón con n_inputs entradas"""
        self.n_inputs = n_inputs
        self.weights = np.random.uniform(-1, 1, n_inputs)  # Pesos aleatorios
        self.bias = np.random.uniform(-1, 1)  # Umbral (bias)
        self.learning_rate = 0.1
        self.errors = []
        self.iteration = 0
        
    def step_function(self, x):
        """Función de activación escalón"""
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """Realiza predicción para una entrada"""
        net_input = np.dot(inputs, self.weights) + self.bias
        return self.step_function(net_input)
    
    def train_pattern(self, inputs, target):
        """Entrena con un patrón usando regla delta"""
        prediction = self.predict(inputs)
        error = target - prediction
        
        # Actualización de pesos usando regla delta
        self.weights += self.learning_rate * error * inputs
        self.bias += self.learning_rate * error
        
        return error
    
    def calculate_error(self, X, y):
        """Calcula el error promedio del conjunto de datos"""
        total_error = 0
        for i in range(len(X)):
            prediction = self.predict(X[i])
            total_error += abs(y[i] - prediction)
        return total_error / len(X)

class PerceptronGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Perceptrón Simple - Implementación Completa")
        self.root.geometry("1200x800")
        
        # Variables del modelo
        self.perceptron = None
        self.X_train = None
        self.y_train = None
        self.dataset_info = {}
        self.training_active = False
        self.error_history = []
        
        self.setup_gui()
        
    def setup_gui(self):
        """Configura la interfaz gráfica"""
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superior para carga de datos y configuración
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sección de carga de dataset
        dataset_frame = ttk.LabelFrame(top_frame, text="1. Carga de Dataset", padding=10)
        dataset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        ttk.Button(dataset_frame, text="Cargar Dataset", 
                  command=self.load_dataset).pack(pady=5)
        
        self.dataset_info_text = tk.Text(dataset_frame, height=8, width=40)
        self.dataset_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Sección de configuración de parámetros
        params_frame = ttk.LabelFrame(top_frame, text="2. Configuración de Parámetros", padding=10)
        params_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Parámetros
        ttk.Label(params_frame, text="Tasa de aprendizaje (η):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.learning_rate_var = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=1, pady=2)
        
        ttk.Label(params_frame, text="Máximo de iteraciones:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.max_iterations_var = tk.StringVar(value="1000")
        ttk.Entry(params_frame, textvariable=self.max_iterations_var, width=10).grid(row=1, column=1, pady=2)
        
        ttk.Label(params_frame, text="Error máximo permitido (ε):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.max_error_var = tk.StringVar(value="0.01")
        ttk.Entry(params_frame, textvariable=self.max_error_var, width=10).grid(row=2, column=1, pady=2)
        
        ttk.Button(params_frame, text="Generar Pesos Aleatorios", 
                  command=self.generate_random_weights).grid(row=3, column=0, columnspan=2, pady=10)
        
        self.weights_text = tk.Text(params_frame, height=4, width=30)
        self.weights_text.grid(row=4, column=0, columnspan=2, pady=5)
        
        # Frame medio para entrenamiento
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sección de entrenamiento
        training_frame = ttk.LabelFrame(middle_frame, text="3. Entrenamiento", padding=10)
        training_frame.pack(fill=tk.X)
        
        button_frame = ttk.Frame(training_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Iniciar Entrenamiento", 
                  command=self.start_training).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Detener Entrenamiento", 
                  command=self.stop_training).pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        self.status_label = ttk.Label(training_frame, text="Estado: Listo para entrenar")
        self.status_label.pack()
        
        # Frame inferior para gráfica y simulación
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sección de gráfica
        graph_frame = ttk.LabelFrame(bottom_frame, text="4. Gráfica de Entrenamiento", padding=10)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Configurar matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_xlabel('Iteraciones')
        self.ax.set_ylabel('Error Promedio')
        self.ax.set_title('Evolución del Error durante el Entrenamiento')
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Sección de simulación
        simulation_frame = ttk.LabelFrame(bottom_frame, text="5. Simulación y Validación", padding=10)
        simulation_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        ttk.Label(simulation_frame, text="Ingresar nuevo patrón:").pack(pady=5)
        
        self.pattern_entries = []
        self.pattern_frame = ttk.Frame(simulation_frame)
        self.pattern_frame.pack(pady=5)
        
        ttk.Button(simulation_frame, text="Predecir", 
                  command=self.predict_pattern).pack(pady=10)
        
        self.prediction_label = ttk.Label(simulation_frame, text="Predicción: -", 
                                        font=('Arial', 12, 'bold'))
        self.prediction_label.pack(pady=10)
        
        ttk.Button(simulation_frame, text="Probar Dataset Completo", 
                  command=self.test_complete_dataset).pack(pady=5)
        
        self.results_text = tk.Text(simulation_frame, height=10, width=30)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def load_dataset(self):
        """Carga el dataset desde archivo"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Dataset",
            filetypes=[
                ("Archivos JSON", "*.json"),
                ("Archivos CSV", "*.csv"),
                ("Archivos Excel", "*.xlsx"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                messagebox.showerror("Error", "Formato de archivo no soportado")
                return
                
            # Detectar automáticamente las columnas
            output_columns = ['salida', 'aprueba', 'output', 'target', 'y']
            output_col = None
            
            for col in output_columns:
                if col in df.columns:
                    output_col = col
                    break
                    
            if output_col is None:
                # Asumir que la última columna es la salida
                output_col = df.columns[-1]
                
            # Separar entradas y salidas
            input_cols = [col for col in df.columns if col != output_col]
            
            self.X_train = df[input_cols].values
            self.y_train = df[output_col].values
            
            # Información del dataset
            n_patterns = len(df)
            n_inputs = len(input_cols)
            n_outputs = 1
            
            self.dataset_info = {
                'patterns': n_patterns,
                'inputs': n_inputs,
                'outputs': n_outputs,
                'input_columns': input_cols,
                'output_column': output_col
            }
            
            # Mostrar información
            info_text = f"Dataset cargado exitosamente:\n\n"
            info_text += f"• Número de patrones: {n_patterns}\n"
            info_text += f"• Número de entradas: {n_inputs}\n"
            info_text += f"• Número de salidas: {n_outputs}\n\n"
            info_text += f"Columnas de entrada:\n"
            for i, col in enumerate(input_cols):
                info_text += f"  {i+1}. {col}\n"
            info_text += f"\nColumna de salida: {output_col}\n\n"
            info_text += "Primeros 5 patrones:\n"
            info_text += str(df.head())
            
            self.dataset_info_text.delete(1.0, tk.END)
            self.dataset_info_text.insert(1.0, info_text)
            
            # Crear perceptrón
            self.perceptron = PerceptronSimple(n_inputs)
            
            # Generar entradas para simulación
            self.create_pattern_entries()
            
            messagebox.showinfo("Éxito", f"Dataset cargado: {n_patterns} patrones, {n_inputs} entradas")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el dataset: {str(e)}")
    
    def create_pattern_entries(self):
        """Crea campos de entrada para simulación de patrones"""
        # Limpiar entradas anteriores
        for widget in self.pattern_frame.winfo_children():
            widget.destroy()
        self.pattern_entries.clear()
        
        if self.dataset_info:
            n_inputs = self.dataset_info['inputs']
            input_cols = self.dataset_info['input_columns']
            
            for i in range(n_inputs):
                ttk.Label(self.pattern_frame, text=f"{input_cols[i]}:").grid(row=i, column=0, sticky=tk.W, pady=2)
                entry = ttk.Entry(self.pattern_frame, width=10)
                entry.grid(row=i, column=1, pady=2)
                self.pattern_entries.append(entry)
    
    def generate_random_weights(self):
        """Genera pesos aleatorios y los muestra"""
        if self.perceptron is None:
            messagebox.showwarning("Advertencia", "Primero carga un dataset")
            return
            
        self.perceptron.weights = np.random.uniform(-1, 1, self.dataset_info['inputs'])
        self.perceptron.bias = np.random.uniform(-1, 1)
        
        weights_text = "Pesos generados:\n\n"
        for i, w in enumerate(self.perceptron.weights):
            weights_text += f"w{i+1} = {w:.4f}\n"
        weights_text += f"θ (umbral/bias) = {self.perceptron.bias:.4f}"
        
        self.weights_text.delete(1.0, tk.END)
        self.weights_text.insert(1.0, weights_text)
    
    def start_training(self):
        """Inicia el entrenamiento del perceptrón"""
        if self.perceptron is None or self.X_train is None:
            messagebox.showwarning("Advertencia", "Primero carga un dataset")
            return
            
        try:
            # Configurar parámetros
            self.perceptron.learning_rate = float(self.learning_rate_var.get())
            max_iterations = int(self.max_iterations_var.get())
            max_error = float(self.max_error_var.get())
            
            # Resetear historial de errores
            self.error_history.clear()
            self.ax.clear()
            self.ax.set_xlabel('Iteraciones')
            self.ax.set_ylabel('Error Promedio')
            self.ax.set_title('Evolución del Error durante el Entrenamiento')
            self.ax.grid(True)
            
            self.training_active = True
            self.progress.start()
            
            # Ejecutar entrenamiento en hilo separado
            training_thread = threading.Thread(
                target=self.training_loop,
                args=(max_iterations, max_error)
            )
            training_thread.daemon = True
            training_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los parámetros: {str(e)}")
    
    def training_loop(self, max_iterations, max_error):
        """Bucle principal de entrenamiento"""
        iteration = 0
        
        while iteration < max_iterations and self.training_active:
            epoch_error = 0
            
            # Entrenar con todos los patrones
            for i in range(len(self.X_train)):
                error = self.perceptron.train_pattern(self.X_train[i], self.y_train[i])
                epoch_error += abs(error)
            
            # Calcular error promedio
            avg_error = epoch_error / len(self.X_train)
            self.error_history.append(avg_error)
            
            # Actualizar gráfica
            self.root.after(0, self.update_plot)
            
            # Actualizar estado
            self.root.after(0, lambda: self.status_label.config(
                text=f"Iteración: {iteration+1}, Error: {avg_error:.6f}"
            ))
            
            # Verificar condición de parada
            if avg_error <= max_error:
                self.root.after(0, lambda: self.training_finished("Convergencia alcanzada"))
                break
                
            iteration += 1
            time.sleep(0.01)  # Pausa pequeña para visualización
        
        if iteration >= max_iterations:
            self.root.after(0, lambda: self.training_finished("Máximo de iteraciones alcanzado"))
    
    def update_plot(self):
        """Actualiza la gráfica de entrenamiento"""
        if self.error_history:
            self.ax.clear()
            self.ax.plot(self.error_history, 'b-', linewidth=2)
            self.ax.set_xlabel('Iteraciones')
            self.ax.set_ylabel('Error Promedio')
            self.ax.set_title('Evolución del Error durante el Entrenamiento')
            self.ax.grid(True)
            self.canvas.draw()
    
    def training_finished(self, reason):
        """Finaliza el entrenamiento"""
        self.training_active = False
        self.progress.stop()
        self.status_label.config(text=f"Entrenamiento finalizado: {reason}")
        
        # Mostrar pesos finales
        weights_text = "Pesos finales:\n\n"
        for i, w in enumerate(self.perceptron.weights):
            weights_text += f"w{i+1} = {w:.4f}\n"
        weights_text += f"θ (umbral/bias) = {self.perceptron.bias:.4f}\n\n"
        weights_text += f"Iteraciones: {len(self.error_history)}\n"
        weights_text += f"Error final: {self.error_history[-1]:.6f}"
        
        self.weights_text.delete(1.0, tk.END)
        self.weights_text.insert(1.0, weights_text)
    
    def stop_training(self):
        """Detiene el entrenamiento"""
        self.training_active = False
    
    def predict_pattern(self):
        """Predice un patrón ingresado por el usuario"""
        if self.perceptron is None:
            messagebox.showwarning("Advertencia", "Primero entrena el perceptrón")
            return
            
        try:
            # Obtener valores de entrada
            inputs = []
            for entry in self.pattern_entries:
                value = entry.get().strip()
                if value == "":
                    messagebox.showwarning("Advertencia", "Completa todos los campos")
                    return
                inputs.append(float(value))
            
            inputs = np.array(inputs)
            
            # Realizar predicción
            prediction = self.perceptron.predict(inputs)
            net_input = np.dot(inputs, self.perceptron.weights) + self.perceptron.bias
            
            self.prediction_label.config(
                text=f"Predicción: {prediction}\n(Net input: {net_input:.4f})"
            )
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los datos de entrada: {str(e)}")
    
    def test_complete_dataset(self):
        """Prueba el perceptrón con todo el dataset"""
        if self.perceptron is None or self.X_train is None:
            messagebox.showwarning("Advertencia", "Primero carga y entrena el perceptrón")
            return
            
        results_text = "Resultados del dataset completo:\n\n"
        correct = 0
        
        for i in range(len(self.X_train)):
            inputs = self.X_train[i]
            expected = self.y_train[i]
            predicted = self.perceptron.predict(inputs)
            
            status = "✓" if predicted == expected else "✗"
            results_text += f"Patrón {i+1}: {status} Pred={predicted}, Esp={expected}\n"
            
            if predicted == expected:
                correct += 1
        
        accuracy = (correct / len(self.X_train)) * 100
        results_text += f"\nPrecisión: {accuracy:.2f}% ({correct}/{len(self.X_train)})"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

def main():
    root = tk.Tk()
    app = PerceptronGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()