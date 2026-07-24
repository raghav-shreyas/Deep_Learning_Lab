Place your 10 EPS output images here, exported at 600 dpi:

fig1_histograms.eps
fig2_correlation_heatmap.eps
fig3_scatter.eps
fig4_boxplots.eps
fig5_training_error.eps
fig6_weight_evolution.eps
fig7_bias_evolution.eps
fig8_confusion_matrix.eps
fig9_lr_comparison.eps
fig10_decision_boundary.eps

Example export command (matplotlib):
    plt.savefig("images/fig1_histograms.eps", format="eps", dpi=600)

Then in Experiment_1_Report.tex, for each figure:
  - uncomment the \includegraphics line
  - delete/comment the \imageplaceholder line below it
