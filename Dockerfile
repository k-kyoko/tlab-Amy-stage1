FROM jupyter/datascience-notebook:lab-4.0.7

USER root

RUN mamba install -y -c conda-forge \
      "python=3.11.*" \
      "numpy=1.26.*" \
      "pandas=2.2.*" \
      "scipy=1.11.*" \
      "scikit-learn=1.4.*" \
      "matplotlib=3.8.*" \
      "seaborn=0.13.*" \
      "pot" \
      "giotto-tda" \
      "pytest=8.*" \
      "pytest-cov=4.*" \
      "jupyter_server=2.8.*" \
      "jupyterlab=4.*" \
      "jupyterlab-git=0.50.*" \
      "jupyterlab-lsp=5.*" \
      "python-lsp-server=1.13.*" \
      "adjusttext=1.3.*" \
    && mamba clean -afy

# jupyterlab-git / jupyterlab-lsp のサーバー拡張を有効化
RUN jupyter server extension enable --py jupyterlab_git && \
    jupyter server extension enable --py jupyter_lsp

USER ${NB_UID}

WORKDIR /home/jovyan/work

CMD ["start-notebook.sh", "--ServerApp.ip='0.0.0.0'", "--ServerApp.port=8889", "--ServerApp.no_browser=True"]
