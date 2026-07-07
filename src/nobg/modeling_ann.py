from torch import nn
from dataclasses import dataclass
from typing import Optional

from .mixin import Revised_Mixin
from .utils import model_card_template


@dataclass
class ANNConfig:
    a: int = 2
    b: int = 1


class ANN(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    tags=["nobg", "ann"],
    model_card_template=model_card_template(class_name="ANN", default_repo="nobg/ann"),
):
    """an AI model for visual question answering"""

    def __init__(self, config: Optional[ANNConfig] = None):
        super().__init__()
        self.config = config or ANNConfig()
        self.layer = nn.Linear(self.config.a, self.config.b, bias=False)

    def forward(self, input_ids):
        return self.layer(input_ids)
