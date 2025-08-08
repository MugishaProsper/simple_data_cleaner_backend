import os
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
import matplotlib
import logging

# Set the backend to 'Agg' to avoid GUI issues in server environments
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    A class for creating various data visualizations from pandas DataFrames.
    Handles figure creation, styling, and saving of plots.
    """
    
    def __init__(self):
        """Initialize the DataVisualizer with default settings."""
        # Create plots directory if it doesn't exist
        self.plots_dir = os.path.join('static', 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set default style
        self._set_plot_style()
    
    def _set_plot_style(self) -> None:
        """Set the default style for all plots."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        
        # Set default color palette
        self.palette = sns.color_palette("husl", 8)
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['grid.alpha'] = 0.3

    def _save_plot(self, filename: Optional[str] = None) -> str:
        """
        Save the current matplotlib figure and return the file path.
        
        Args:
            filename: Optional custom filename (without extension)
            
        Returns:
            str: Relative path to the saved plot file
        """
        if not filename:
            filename = f"plot_{uuid.uuid4().hex[:8]}"
        
        # Ensure filename is safe and has the right extension
        safe_filename = f"{''.join(c if c.isalnum() or c in '-_' else '_' for c in filename)}.png"
        filepath = os.path.join(self.plots_dir, safe_filename)
        
        try:
            plt.tight_layout()
            plt.savefig(filepath, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            return f"static/plots/{safe_filename}"
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            plt.close()
            raise
    
    def create_visualization(self, df: pd.DataFrame, plot_type: str, 
                           x: Optional[str] = None, y: Optional[str] = None, hue: Optional[str] = None,
                           title: Optional[str] = None, file_id: Optional[str] = None) -> str:
        """
        Create a visualization based on the specified type and parameters.
        
        Args:
            df: Input DataFrame
            plot_type: Type of plot to create (line, bar, scatter, hist, box, violin, pairplot, heatmap)
            x: Column name for x-axis
            y: Column name for y-axis (optional for some plot types)
            hue: Column name for color encoding (optional)
            title: Plot title (optional)
            file_id: Unique identifier for the file (used in filename)
            
        Returns:
            str: Path to the saved plot file
        """
        plot_handlers = {
            'line': self._create_line_plot,
            'bar': self._create_bar_plot,
            'scatter': self._create_scatter_plot,
            'hist': self._create_histogram,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'pairplot': self._create_pair_plot,
            'heatmap': self._create_correlation_heatmap
        }
        
        if plot_type not in plot_handlers:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        try:
            # Generate a filename based on plot type and columns
            filename = f"{plot_type}_{file_id or uuid.uuid4().hex[:8]}"
            if x:
                filename += f"_x_{x}"
            if y:
                filename += f"_y_{y}"
            
            # Call the appropriate plot handler
            if plot_type == 'line':
                return self._create_line_plot(df, x or "", y or "", hue, title, filename)
            elif plot_type == 'bar':
                return self._create_bar_plot(df, x or "", y or "", hue, title, filename)
            elif plot_type == 'scatter':
                return self._create_scatter_plot(df, x or "", y or "", hue, title, filename)
            elif plot_type == 'hist':
                return self._create_histogram(df, x or "", y, hue, title, filename)
            elif plot_type == 'box':
                return self._create_box_plot(df, x or "", y, hue, title, filename)
            elif plot_type == 'violin':
                return self._create_violin_plot(df, x or "", y, hue, title, filename)
            elif plot_type == 'pairplot':
                return self._create_pair_plot(df, x, y, hue, title, filename)
            elif plot_type == 'heatmap':
                return self._create_correlation_heatmap(df, x, y, hue, title, filename)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
        except Exception as e:
            logger.error(f"Error creating {plot_type} plot: {e}")
            raise
    
    def _create_line_plot(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                         title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a line plot."""
        plt.figure(figsize=(12, 6))
        
        if hue and hue in df.columns:
            sns.lineplot(data=df, x=x, y=y, hue=hue, palette=self.palette)
        else:
            sns.lineplot(data=df, x=x, y=y, color=self.palette[0])
        
        plt.title(title or f"{y} over {x}")
        plt.grid(True, alpha=0.3)
        
        if df[x].nunique() > 10:  # Rotate x-labels if many x-values
            plt.xticks(rotation=45, ha='right')
        
        return self._save_plot(filename)
    
    def _create_bar_plot(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                        title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a bar plot."""
        plt.figure(figsize=(12, 6))
        
        # If too many categories, show only top N
        if df[x].nunique() > 20:
            top_n = df.groupby(x)[y].mean().nlargest(20).index
            df = df[df[x].isin(top_n)]
        
        if hue and hue in df.columns:
            sns.barplot(data=df, x=x, y=y, hue=hue, palette=self.palette)
        else:
            sns.barplot(data=df, x=x, y=y, color=self.palette[0])
        
        plt.title(title or f"{y} by {x}")
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        return self._save_plot(filename)
    
    def _create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None,
                           title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a scatter plot."""
        plt.figure(figsize=(10, 8))
        
        if hue and hue in df.columns:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, palette=self.palette, alpha=0.7)
        else:
            sns.scatterplot(data=df, x=x, y=y, color=self.palette[0], alpha=0.7)
        
        plt.title(title or f"{y} vs {x}")
        plt.grid(True, alpha=0.3)
        
        return self._save_plot(filename)
    
    def _create_histogram(self, df: pd.DataFrame, x: str, y: Optional[str] = None,
                         hue: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a histogram or distribution plot."""
        plt.figure(figsize=(10, 6))
        
        if hue and hue in df.columns:
            for i, (category, group) in enumerate(df.groupby(hue)):
                sns.histplot(data=group, x=x, color=self.palette[i % len(self.palette)], 
                            alpha=0.5, label=str(category), kde=True)
            plt.legend()
        else:
            sns.histplot(data=df, x=x, color=self.palette[0], kde=True)
        
        plt.title(title or f"Distribution of {x}")
        plt.grid(True, alpha=0.3)
        
        return self._save_plot(filename)
    
    def _create_box_plot(self, df: pd.DataFrame, x: str, y: Optional[str] = None,
                        hue: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a box plot."""
        plt.figure(figsize=(12, 6))
        
        if y:  # Box plot with x and y specified
            if hue and hue in df.columns:
                sns.boxplot(data=df, x=x, y=y, hue=hue, palette=self.palette)
            else:
                sns.boxplot(data=df, x=x, y=y, color=self.palette[0])
            plt.title(title or f"Box plot of {y} by {x}")
        else:  # Single variable box plot
            sns.boxplot(data=df, x=x, color=self.palette[0])
            plt.title(title or f"Box plot of {x}")
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        return self._save_plot(filename)
    
    def _create_violin_plot(self, df: pd.DataFrame, x: str, y: Optional[str] = None,
                           hue: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a violin plot."""
        plt.figure(figsize=(12, 6))
        
        if y:  # Violin plot with x and y specified
            if hue and hue in df.columns:
                sns.violinplot(data=df, x=x, y=y, hue=hue, palette=self.palette, split=True)
            else:
                sns.violinplot(data=df, x=x, y=y, color=self.palette[0])
            plt.title(title or f"Violin plot of {y} by {x}")
        else:  # Single variable violin plot
            sns.violinplot(data=df, x=x, color=self.palette[0])
            plt.title(title or f"Violin plot of {x}")
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        return self._save_plot(filename)
    
    def _create_pair_plot(self, df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                         hue: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a pair plot (scatter matrix) of variables."""
        # Select only numeric columns for pair plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            raise ValueError("No numeric columns available for pair plot")
        
        # Limit the number of columns to avoid huge plots
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]
            logger.warning("Limiting pair plot to first 5 numeric columns")
        
        plot_df = df[numeric_cols]
        
        # Add hue if specified and it's a categorical column
        if hue and hue in df.columns and df[hue].nunique() < 10:  # Limit to 10 categories
            plot_df[hue] = df[hue]
            g = sns.pairplot(plot_df, hue=hue, palette=self.palette, diag_kind='kde')
        else:
            g = sns.pairplot(plot_df, palette=self.palette, diag_kind='kde')
        
        # Set title if provided
        if title:
            g.fig.suptitle(title, y=1.02)
        
        # Save the plot
        if not filename:
            filename = f"pairplot_{uuid.uuid4().hex[:8]}"
        
        filepath = os.path.join(self.plots_dir, f"{filename}.png")
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return f"static/plots/{filename}.png"
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None,
                                  hue: Optional[str] = None, title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """Create a correlation heatmap of numeric variables."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation heatmap")
        
        # Calculate correlation matrix
        corr = numeric_df.corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Set up the matplotlib figure
        plt.figure(figsize=(12, 10))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot=True, fmt=".2f")
        
        plt.title(title or "Correlation Heatmap", pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        return self._save_plot(filename or "correlation_heatmap")