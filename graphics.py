# ===================================================================================================
# HIGH-DENSITY VISUAL LAYOUT: MULTI-BRANCH BLIND-SPOT MAPPING FOR IEEE JOURNALS (GIANT TEXT EDITION)
# SPECIFICATIONS: THREE CONCURRENT COMPACT RECEPTIVE FIELDS WITH LARGE LEGIBLE TYPOGRAPHY
# ===================================================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Configure professional sans-serif typography for peer-review presentation
plt.rcParams['font.family'] = 'sans-serif'

# Initialize a wide canvas accommodating three distinct spatial branches
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6.5), dpi=300)
grid_size = 5
center = grid_size // 2

def draw_base_grid(ax, title_text):
    """Draws a clean boundary grid with a locked aspect ratio and hidden axes."""
    for x in range(grid_size + 1):
        ax.axhline(x, color='#E2E6EA', linestyle='-', linewidth=1.5, zorder=1)
        ax.axvline(x, color='#E2E6EA', linestyle='-', linewidth=1.5, zorder=1)
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=18, color='#212529')

# ---------------------------------------------------------------------------------------------------
# BRANCH A: LOCAL CONTINUOUS 3x3 VECINDARIO
# ---------------------------------------------------------------------------------------------------
draw_base_grid(ax1, "Branch A: Ultra-Local Context\n(Dense 3$\\times$3 Convolution)")

# Paint the immediate 3x3 neighborhood (excluding center)
for dy in [-1, 0, 1]:
    for dx in [-1, 0, 1]:
        if dy == 0 and dx == 0: continue
        ax1.add_patch(patches.Rectangle((center + dx, center + dy), 1, 1, 
                                        facecolor='#E6F2FF', edgecolor='#0066CC', alpha=0.8, lw=2.5, zorder=2))

# ---------------------------------------------------------------------------------------------------
# BRANCH B: GLOBAL EXTENDED 5x5 VECINDARIO
# ---------------------------------------------------------------------------------------------------
draw_base_grid(ax2, "Branch B: Meso-Scale Global Context\n(Full 5$\\times$5 Convolution)")

# Paint the entire 5x5 area (excluding center)
for dy in range(-2, 3):
    for dx in range(-2, 3):
        if dy == 0 and dx == 0: continue
        ax2.add_patch(patches.Rectangle((center + dx, center + dy), 1, 1, 
                                        facecolor='#FFE6E6', edgecolor='#CC0000', alpha=0.6, lw=2.5, zorder=2))

# ---------------------------------------------------------------------------------------------------
# BRANCH C: DILATED ATROUS VECINDARIO (DILATION=2)
# ---------------------------------------------------------------------------------------------------
draw_base_grid(ax3, "Branch C: Macro-Scale Sparse Context\n(Atrous 3$\\times$3 Convolution, $d=2$)")

# Paint saltatory pixels dictated by dilation = 2
for dy in [-2, 0, 2]:
    for dx in [-2, 0, 2]:
        if dy == 0 and dx == 0: continue
        ax3.add_patch(patches.Rectangle((center + dx, center + dy), 1, 1, 
                                        facecolor='#E6FFE6', edgecolor='#009933', alpha=0.8, lw=2.5, zorder=2))

# ---------------------------------------------------------------------------------------------------
# UNIVERSAL ANCHOR: ENFORCE THE MATRICIAL BLIND-SPOT ACROSS ALL EXPERIMENTAL STREAMS
# ---------------------------------------------------------------------------------------------------
for ax in [ax1, ax2, ax3]:
    # Hard lock the geometric core index
    blind_spot = patches.Rectangle((center, center), 1, 1, facecolor='#212529', edgecolor='#000000', lw=3.0, zorder=3)
    ax.add_patch(blind_spot)
    
    # CORREGIDO: Ajuste estricto de linespacing y empaquetado interno de la notación matemática
    ax.text(center + 0.5, center + 0.5, 'Strict\nBlind\nSpot\n$\\mathbf{\\omega_0=0}$', 
            color='white', ha='center', va='center', fontsize=12, fontweight='bold', linespacing=0.85, zorder=4)

# Save vector-grade visual panel directly to workspace disk
plt.tight_layout(pad=3.0)
output_diagram_name = "fig_multi_branch_diagram_a.png"
plt.savefig(output_diagram_name, bbox_inches='tight', dpi=300)
print(f"-> [SUCCESS] Peer-review diagram successfully exported to: '{output_diagram_name}'")
plt.show()

# ===================================================================================================
# HARDWARE BUS AND COMPACT AUDIT FLOW CLOSURE (SYNTHETIC VERIFICATION EDITION)
# ===================================================================================================
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Canvas configuration optimized for massive visibility in IEEE templates
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(16.5, 9.0), dpi=300) # Slightly taller canvas to close the NO loop cleanly

# =====================================================================
# 1. RIGID COORDINATE HARDWARE PATCH MAPPING & ARCHITECTURAL STREAMS
# =====================================================================
y_positions = [4.5, 3.2, 1.9, 0.6]
labels_inputs = [
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{Local}) concrete}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{Global}) concrete}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{Dilated}) concrete}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{Boundary}) concrete}$"
]

for y, l_in in zip(y_positions, labels_inputs):
    
    box_in = patches.FancyBboxPatch((0.0, y - 0.28), 2.25, 0.56, boxstyle="round,pad=0.03",
                                    facecolor="#F8F9FA", edgecolor="#495057", lw=3.5, zorder=3)
    ax.add_patch(box_in)
    
    ax.text(1.125, y, l_in, ha="center", va="center", fontsize=15, fontweight="bold", zorder=4)
    
    
    box_conv1x1 = patches.Rectangle((3.0, y - 0.28), 1.4, 0.56, facecolor="#E6F2FF", edgecolor="#0066CC", lw=3.5, zorder=3)
    ax.add_patch(box_conv1x1)
    ax.text(3.7, y, r"$\mathrm{Conv2D}_{1 \times 1}$", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    
    
    ax.annotate("", xy=(3.0, y), xytext=(2.35, y),
                arrowprops=dict(arrowstyle="-|>", color="#495057", lw=3.0, mutation_scale=16), zorder=4)


box_concat = patches.Rectangle((5.0, 0.28), 2.1, 4.8, facecolor="#E9ECEF", edgecolor="#343A40", lw=3.5, zorder=3)
ax.add_patch(box_concat)
box_concat_text = "Score\nConcatenation\nBlock\n\n" + r"$\mathbf{F}_{\mathrm{concat}}$"
ax.text(6.05, 2.68, box_concat_text, ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)


box_softmax = patches.Rectangle((7.8, 2.0), 2.5, 1.3, facecolor="#FFE6E6", edgecolor="#CC0000", lw=3.5, zorder=3)
ax.add_patch(box_softmax)
box_softmax_text = "Spatial Softmax\n\n" + r"$[\mathbf{W}_1, \dots, \mathbf{W}_M]$"
ax.text(9.05, 2.65, box_softmax_text, ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)


circle_op = patches.Circle((11.4, 2.65), 0.45, facecolor="#E6FFE6", edgecolor="#009933", lw=3.5, zorder=3)
ax.add_patch(circle_op)
ax.text(11.4, 2.65, r"$\sum \odot$", ha="center", va="center", fontsize=24, fontweight="bold", zorder=4)
ax.text(11.4, 1.8, "Weighted\nAggregation", ha="center", va="top", fontsize=12, color="#006622", fontweight="bold", zorder=4)


box_output = patches.FancyBboxPatch((13.3, 2.0), 2.4, 1.3, boxstyle="round,pad=0.03",
                                     facecolor="#FFF2E6", edgecolor="#FF8000", lw=3.5, zorder=3)
ax.add_patch(box_output)
ax.text(14.5, 2.65, "Fused Tensor\n\n" + r"$\mathbf{F}_{\mathrm{unified}}$", 
        ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)

# =====================================================================
# 2. ONLINE GRADIENT CHECKING & AUTOGRAD LOOP
# =====================================================================
# Autograd Engine alineado a X=12.4
box_autograd = patches.FancyBboxPatch((12.4, -1.8), 2.8, 1.3, boxstyle="round,pad=0.04",
                                      facecolor="#FFF9E6", edgecolor="#D39E00", lw=4.0, zorder=3)
ax.add_patch(box_autograd)
autograd_text = "Analytical Autograd\nEngine\n\n" + r"$\nabla_{\mathbf{Y}_c} \hat{\mathbf{X}}_c = \mathrm{backward}(\hat{x}_c)$"
ax.text(13.8, -1.15, autograd_text, ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)

# CORREGIDO: Rombo de decisión alineado en su eje central X=8.65
diamond_switch = patches.Polygon([[8.65, -0.5], [9.9, -1.15], [8.65, -1.8], [7.4, -1.15]], 
                                 facecolor="#E2E3E5", edgecolor="#383D41", lw=3.5, zorder=3)
ax.add_patch(diamond_switch)
ax.text(8.65, -1.15, r"$\nabla_{\mathbf{Y}_c} \hat{\mathbf{X}}_c \equiv 0.0?$", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

# CORREGIDO: Bloque HALT alineado a X=7.45
box_halt = patches.FancyBboxPatch((7.45, -3.5), 2.4, 1.1, boxstyle="round,pad=0.03",
                                    facecolor="#FFF5F5", edgecolor="#DC3545", lw=3.5, zorder=3)
ax.add_patch(box_halt)
ax.text(8.65, -2.95, "HALT EXECUTION\nGradient Leakage\nException Raised", color="#DC3545",
        ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

# Labels adjustement
ax.text(7.0, -0.9, "YES", color="green", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
ax.text(8.9, -2.0, "NO", color="red", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

# =====================================================================
# 3. INTER-BLOCK VECTOR STREAM ROUTING (PRECISE BOUNDS)
# =====================================================================
targets_concat_y = [3.8, 2.9, 2.1, 1.2]
for y_in, y_tar in zip(y_positions, targets_concat_y):
    ax.plot([4.4, 4.7, 4.7], [y_in, y_in, y_tar], color="#0066CC", lw=2.5, zorder=4)
    ax.annotate("", xy=(5.0, y_tar), xytext=(4.7, y_tar),
                arrowprops=dict(arrowstyle="-|>", color="#0066CC", lw=2.5, mutation_scale=16), zorder=4)

ax.plot([7.1, 7.8], [2.65, 2.65], color="#343A40", lw=3.0, zorder=2)
ax.plot([10.3, 10.95], [2.65, 2.65], color="#CC0000", lw=3.0, zorder=2)
ax.plot([11.85, 13.3], [2.65, 2.65], color="#009933", lw=3.0, zorder=2)

ax.annotate("", xy=(7.8, 2.65), xytext=(7.5, 2.65), arrowprops=dict(arrowstyle="-|>", color="#343A40", lw=3.0, mutation_scale=18), zorder=4)
ax.annotate("", xy=(10.95, 2.65), xytext=(10.6, 2.65), arrowprops=dict(arrowstyle="-|>", color="#CC0000", lw=3.0, mutation_scale=18), zorder=4)
ax.annotate("", xy=(13.3, 2.65), xytext=(12.7, 2.65), arrowprops=dict(arrowstyle="-|>", color="#009933", lw=3.0, mutation_scale=18), zorder=4)

# =====================================================================
# 4. HERMETIC PRES-BUS CON TRAZADO AEREO INTEGRAL
# =====================================================================
for y in y_positions:
    ax.plot([0.0, -0.25], [y, y], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([-0.25, -0.25], [0.6, 5.5], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([-0.25, 11.4], [5.5, 5.5], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([11.4, 11.4], [5.5, 3.1], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.annotate("", xy=(11.4, 3.1), xytext=(11.4, 3.2), arrowprops=dict(arrowstyle="-|>", color="#495057", lw=2.5, mutation_scale=18), zorder=4)

ax.text(5.7, 5.8, r"Hermetic Feature Preservation Bus ($\mathbf{F}_{\Omega}$)", ha="center", va="center", fontsize=15, color="#212529", fontweight="bold", zorder=4)

# =====================================================================
# 5. CLOSING THE FEEDBACK VALIDATION LOOPS
# =====================================================================
# ORANGE BUS
ax.plot([15.7, 16.1, 16.1, 15.2], [2.65, 2.65, -1.15, -1.15], color="#FF8000", lw=3.0, zorder=1)
ax.annotate("", xy=(15.2, -1.15), xytext=(15.3, -1.15), arrowprops=dict(arrowstyle="-|>", color="#FF8000", lw=3.0, mutation_scale=18), zorder=4)
ax.text(16.2, -0.4, "Isolate Central\nPrediction " + r"$\hat{x}_c$", ha="left", va="center", fontsize=12, color="#FF8000", fontweight="bold", zorder=4)

# YELLOW BUS 
ax.annotate("", xy=(9.9, -1.15), xytext=(12.4, -1.15), arrowprops=dict(arrowstyle="-|>", color="#D39E00", lw=3.0, mutation_scale=20), zorder=4)

# GREEN LINK (YES)
ax.plot([7.4, 6.05, 6.05], [-1.15, -1.15, 0.28], color="green", lw=3.0, zorder=1)
ax.annotate("", xy=(6.05, 0.28), xytext=(6.05, 0.1), arrowprops=dict(arrowstyle="-|>", color="green", lw=3.0, mutation_scale=16), zorder=4)
ax.text(4.5, -0.8, "PROCEED:\nBackpropagation\n& Weight Update", color="green", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

# RED ARROW (NO)
ax.annotate("", xy=(8.65, -2.4), xytext=(8.65, -1.8), arrowprops=dict(arrowstyle="-|>", color="red", lw=3.0, mutation_scale=18), zorder=4)

# Border adjustement
ax.set_xlim(-0.8, 18.5)
ax.set_ylim(-3.8, 6.2)
ax.axis('off')

plt.tight_layout()
output_image_name = 'self_supervised_bsn_verification_flow_a.png'
plt.savefig(output_image_name, bbox_inches='tight', dpi=300)
plt.show()

print(f"Geometry completed. Flow diagram successfully preserved at: '{output_image_name}'")
