from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from medai.main import runthrough_api


app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
#app.state.model = load_model()

@app.get("/")
def root():
    return {
    'greeting': 'Hello, this is the first test'
    }



@app.get('/diagnosis')
def get_diagnosis(inputs: str):
    output = runthrough_api(inputs) # adapt function??
    return output
