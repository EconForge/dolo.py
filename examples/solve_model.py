"""Simple example demonstrating how to load and solve a model with pretty Rich output."""

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import dolo

console = Console()

# 1. Header and Model Loading
console.print(
    Panel.fit(
        "[bold cyan]Dolo.py[/bold cyan] [white]— Dynamic Stochastic Model Solver[/white]\n"
        "[dim]Demonstrating Time Iteration, Acceleration, and Newton-Krylov algorithms[/dim]",
        box=box.ROUNDED,
        border_style="cyan",
    )
)

model_path = Path(__file__).parent / "models" / "rbc.yaml"
model = dolo.yaml_import(str(model_path))

# Model Summary Table
model_info = Table(title="[bold]Model Specification[/bold]", box=box.ROUNDED)
model_info.add_column("Property", style="bold")
model_info.add_column("Details", style="cyan")

model_info.add_row("Model Name", model.name if hasattr(model, "name") else "RBC")
model_info.add_row("States", ", ".join(model.symbols["states"]))
model_info.add_row("Controls", ", ".join(model.symbols["controls"]))
model_info.add_row("Exogenous", ", ".join(model.symbols["exogenous"]))

console.print(model_info)
console.print()

results_summary = Table(
    title="[bold green]Solver Comparison[/bold green]", box=box.ROUNDED
)
results_summary.add_column("Algorithm", style="bold")
results_summary.add_column("Converged", justify="center")
results_summary.add_column("Iterations", justify="right", style="cyan")
results_summary.add_column("Residual Error (eps)", justify="right", style="magenta")
results_summary.add_column("Step Error (eta)", justify="right", style="magenta")

# 2. Standard Time Iteration
console.rule("[bold cyan]1. Standard Time Iteration[/bold cyan]")
sol_ti = dolo.time_iteration(model, verbose=True)
results_summary.add_row(
    "Standard Time Iteration",
    "[green]✓[/green]" if sol_ti.converged else "[red]✗[/red]",
    str(sol_ti.iterations),
    f"{sol_ti.error_res:.2e}" if sol_ti.error_res else "—",
    f"{sol_ti.error_step:.2e}",
)

# 3. Accelerated Time Iteration
console.rule("[bold cyan]2. Accelerated Time Iteration (accelerate=True)[/bold cyan]")
sol_acc = dolo.time_iteration(model, accelerate=True, verbose=True)
results_summary.add_row(
    "Accelerated TI (Neumann)",
    "[green]✓[/green]" if sol_acc.converged else "[red]✗[/red]",
    str(sol_acc.iterations),
    f"{sol_acc.error_res:.2e}" if sol_acc.error_res else "—",
    f"{sol_acc.error_step:.2e}",
)

# 4. Newton-Krylov Policy Solver
console.rule("[bold cyan]3. Newton-Krylov Solver (linsolve='gmres')[/bold cyan]")
sol_newton = dolo.newton(
    model, linsolve="gmres", linesearch="backtracking", verbose=True
)
results_summary.add_row(
    "Newton-Krylov (GMRES)",
    "[green]✓[/green]" if sol_newton.converged else "[red]✗[/red]",
    str(sol_newton.iterations),
    f"{sol_newton.error_res:.2e}",
    f"{sol_newton.error_step:.2e}",
)

# 5. Print Comparison Summary
console.print()
console.print(results_summary)
console.print()

# 6. Pretty Print Policy Function at Steady State
dr = sol_newton.dr
s0 = model.calibration["states"]
m0 = model.calibration["exogenous"]
controls_at_ss = dr(m0, s0)

policy_table = Table(
    title="[bold yellow]Decision Rule at Steady State[/bold yellow]", box=box.ROUNDED
)
policy_table.add_column("Control Variable", style="bold")
policy_table.add_column("Steady State Value", justify="right", style="green")

for name, val in zip(model.symbols["controls"], controls_at_ss):
    policy_table.add_row(name, f"{val:.4f}")

console.print(policy_table)
console.print()
