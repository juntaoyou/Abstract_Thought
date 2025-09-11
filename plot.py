import matplotlib.pyplot as plt
import torch
from utils.plot import plot0, plot1, plot2, plot3, load


base_file_path = "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.base"
file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.BB"]

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA"]

labels = ["Expertise_A or Informativeness_A",
          "Expertise_Informativeness_AA",
          "Expertise_A",
          "Informativeness_A"]
# plot2(base_file_path, file_paths, title="Expertise_A_Informativeness_A", labels=labels)

file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.BB"]

labels = ["Expertise_B or Informativeness_B",
          "Expertise_Informativeness_BB",
          "Expertise_B",
          "Informativeness_B"]
# plot2(base_file_path, file_paths, title="Expertise_B_Informativeness_B", labels=labels)


file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness.AA"]

labels = ["Expertise_A or Informativeness_A",
          "Expertise_Informativeness_AA",
          "Expertise_A",
          "Informativeness_A"]
# plot2(base_file_path, file_paths, title="Expertise_A_Informativeness_A", labels=labels)

file_paths = [
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness_Style.BA"]

labels = ["Informativeness_B or Style_A",
          "Informativeness_Style_BA",
          "Informativeness_B",
          "Style_A",
          ]
plot2(base_file_path, file_paths, title="Informativeness_B_Style_A", labels=labels)


file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.B",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
              "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Style.BA"]

labels = ["Expertise_B or Style_A",
          "Expertise_Style_BA",
          "Expertise_B",
          "Style_A"]
# plot2(base_file_path, file_paths, title="Expertise_B_Style_A", labels=labels)

# file_paths = ["/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Informativeness.B",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Style.A",
#               "/NAS/yjt/Abstract_Thought/data/activation.train.Qwen3-8B.Expertise_Informativeness_Style.ABA"]

# labels = ["Expertise_A",
#           "Informativeness.B",
#           "Style_A", 
#           "Expertise_A or Informativeness_B or Style_A",
#           "Expertise_Informativeness_Style_ABA"]

# plot3(base_file_path, file_paths, title="Expertise_A_Informativeness_B_Style_A", labels=labels)