from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from MlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)