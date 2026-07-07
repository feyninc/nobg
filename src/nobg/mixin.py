from huggingface_hub import PyTorchModelHubMixin, whoami

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


class Revised_Mixin(PyTorchModelHubMixin):
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: Optional[Union[dict, "DataclassInstance"]] = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool = False,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        delete_patterns: Optional[Union[List[str], str]] = None,
        model_card_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        if model_card_kwargs is None:
            model_card_kwargs = {}
        if "/" not in repo_id:
            username = whoami()["name"]
            repo_id = f"{username}/{repo_id}"
        model_card_kwargs["repo_id"] = repo_id
        return super().push_to_hub(
            repo_id,
            config=config,
            commit_message=commit_message,
            private=private,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            model_card_kwargs=model_card_kwargs,
        )


Revised_Mixin.push_to_hub.__doc__ = PyTorchModelHubMixin.push_to_hub.__doc__
