def print_help_text():
    print("\n")
    print("How to use\n"
          "==========\n\n"
          "Every heatmap is annotated with the name of the (global memory) buffer is visualizes.\n"
          "The heatmap color translates to the number of memory accesses at the location of the color.\n"
          "Refer to the \"Color legend\" for the mapping between color and access count.\n"
          "You can adapt the colormap to any buffer by simply clicking on the accourding heatmap.\n"
          "Please note that this might invalidate the other heatmaps as the the colormap might"
          "not cover their range of access counts.\n"
          "Invalid access counts are then shown as in green. "
          "Press \"r\" on your keyboard to reset the colormap."
          "\n"
          "The \"Access histogram\" provides an overview of the total number of accesses over time.\n"
          "Use the \"Time selection\" slider to change the timerange which is visualized by the heatmaps."
          )
