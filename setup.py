from distutils.core import setup

setup(
    name="compasspathways-sentiment2d",
    packages=["sentiment2d"],
    description="COMPASS Two-dimesional Sentiment Model",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "patsy",
        "torch",
        "transformers",
        "plotly",
        "kaleido",
    ],
)
