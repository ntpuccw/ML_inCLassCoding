"""
GUI version of gar0907.ipynb.

Provides a Tkinter interface to:
1. Load a data file (CSV/Excel) and preview it.
2. Draw a boxplot of the numeric columns.
3. Standardize (z-score) the numeric columns and compare original vs.
   standardized boxplots side by side.
4. Draw a correlation coefficient heatmap.
5. Draw a PCA scree plot and Pareto plot.
6. Draw a heatmap of the first two principal components' loadings.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA


class DataExplorerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Explorer / PCA GUI")
        self.geometry("1100x700")
        self.minsize(900, 600)

        self.df = None
        self.numeric_df = None
        self.normalized_df = None
        self.pca = None

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_layout(self):
        # Use grid so the left panel keeps a fixed width and the plot area
        # gets all remaining space, regardless of preview table content.
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # Left panel: controls + data preview
        left = ttk.Frame(self, padding=10, width=320)
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)

        ttk.Button(left, text="Load Data File", command=self.load_file).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(left, text="Boxplot (Original)", command=self.plot_boxplot).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(
            left,
            text="Standardize + Boxplot Compare",
            command=self.plot_standardized_boxplot,
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            left, text="Correlation Heatmap", command=self.plot_correlation_heatmap
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            left, text="Scree + Pareto Plot", command=self.plot_scree_pareto
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            left, text="PCA Loadings Heatmap (PC1 & PC2)", command=self.plot_pca_heatmap
        ).pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Data Preview:").pack(anchor=tk.W, pady=(10, 0))

        # Wrap the treeview with scrollbars so wide/long tables can't force
        # the left panel (or window) to grow. The container gets a fixed
        # size and grid_propagate(False) so the Treeview's requested width
        # (based on its column widths) is never allowed to bubble up.
        preview_container = ttk.Frame(left, width=300, height=260)
        preview_container.pack(fill=tk.BOTH, expand=True)
        preview_container.grid_propagate(False)
        vsb = ttk.Scrollbar(preview_container, orient="vertical")
        hsb = ttk.Scrollbar(preview_container, orient="horizontal")
        self.preview = ttk.Treeview(
            preview_container,
            height=15,
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.config(command=self.preview.yview)
        hsb.config(command=self.preview.xview)
        preview_container.rowconfigure(0, weight=1)
        preview_container.columnconfigure(0, weight=1)
        self.preview.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        # The hidden "#0" tree column defaults to a non-zero width; remove it
        # so it can't add unexpected width to the widget's size request.
        self.preview.column("#0", width=0, stretch=False)

        # Right panel: plot canvas area
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew")
        self.canvas = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Data files", "*.csv *.xlsx *.xls *.txt"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path, sep=None, engine="python")
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not read file:\n{exc}")
            return

        self.df = df
        self.numeric_df = df.select_dtypes(include="number")
        self.normalized_df = None
        self.pca = None
        self._update_preview()

    def _update_preview(self):
        self.preview.delete(*self.preview.get_children())
        if self.df is None:
            return
        cols = list(self.df.columns)
        self.preview["columns"] = cols
        self.preview["show"] = "headings"
        for c in cols:
            self.preview.heading(c, text=c)
            self.preview.column(c, width=80, stretch=False)
        for _, row in self.df.head(10).iterrows():
            self.preview.insert("", tk.END, values=list(row))

    def _require_data(self):
        if self.df is None or self.numeric_df.empty:
            messagebox.showwarning("No Data", "Please load a data file first.")
            return False
        return True

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    def _show_figure(self, fig):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def plot_boxplot(self):
        if not self._require_data():
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        self.numeric_df.plot(kind="box", vert=False, ax=ax)
        ax.set_title("Distribution of Data")
        ax.grid(True)
        fig.tight_layout()
        self._show_figure(fig)

    def plot_standardized_boxplot(self):
        if not self._require_data():
            return
        self.normalized_df = (
            self.numeric_df - self.numeric_df.mean()
        ) / self.numeric_df.std()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.numeric_df.plot(kind="box", vert=False, ax=axes[0])
        axes[0].set_title("Original Data")
        self.normalized_df.plot(kind="box", vert=False, ax=axes[1])
        axes[1].set_title("Standardized Data")
        fig.tight_layout()
        self._show_figure(fig)

    def plot_correlation_heatmap(self):
        if not self._require_data():
            return
        corr = self.numeric_df.corr()

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(
                    j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8
                )
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        self._show_figure(fig)

    def _ensure_normalized(self):
        if self.normalized_df is None:
            self.normalized_df = (
                self.numeric_df - self.numeric_df.mean()
            ) / self.numeric_df.std()

    def plot_scree_pareto(self):
        if not self._require_data():
            return
        self._ensure_normalized()

        self.pca = PCA()
        self.pca.fit(self.normalized_df)
        explained_var = self.pca.explained_variance_ratio_ * 100
        cumulative_var = np.cumsum(explained_var)
        components = range(1, len(explained_var) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(components, explained_var, marker="o")
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Explained Variance (%)")
        axes[0].set_title("Scree Plot")
        axes[0].set_xticks(list(components))

        axes[1].bar(components, explained_var)
        axes[1].plot(components, cumulative_var, color="red", marker="o")
        axes[1].set_xlabel("Principal Component")
        axes[1].set_ylabel("Variance (%)")
        axes[1].set_title("Pareto Plot")
        axes[1].set_xticks(list(components))

        fig.tight_layout()
        self._show_figure(fig)

    def plot_pca_heatmap(self):
        if not self._require_data():
            return
        self._ensure_normalized()

        if self.pca is None:
            self.pca = PCA()
            self.pca.fit(self.normalized_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        feature_names = self.normalized_df.columns
        sns.heatmap(
            self.pca.components_[:2],
            annot=True,
            cmap="coolwarm",
            center=0,
            xticklabels=feature_names,
            yticklabels=["PC1", "PC2"],
            ax=ax,
        )
        ax.set_title("Heatmap of First Two Principal Components")
        ax.set_xlabel("Features")
        ax.set_ylabel("Principal Components")
        fig.tight_layout()
        self._show_figure(fig)


if __name__ == "__main__":
    app = DataExplorerApp()
    app.mainloop()
