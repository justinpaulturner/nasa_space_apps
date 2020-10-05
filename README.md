# Fire Detection For School Evacuation Notices

### Summary
This project is a public Flask web app that identifies fires from a real-time satellite data feed and gives evacuation notices for the nearby schools of the identified fires.

### How We Developed This Project
The impacts and loss of life from the 2020 west coast wildfires inspired our team to develop a tool that schools can monitor to check if they need to evacuate. We approached this project by setting 5 high-level goals and completing each incrementally, step-by-step.

In a Python development environment we trained a neural network on wildfire satellite data and achieved 90% accuracy. We integrated the trained model into a Flask web app that runs a live feed of satellite imagery. When a wildfire is detected, the image of the map is displayed with its location. Schools that are within 50 miles of the identified fires will also display with the identified fire.

We encountered many problems while training the neural network, such as file compatibility and infrared data integration into the model.

### How We Used Space Agency Data in This Project
We used labled Biomass Burning Smoke data to train our model and the neural network can ingest satelite data from NOAA Geostationary Operational Environmental Satellites (GOES) 16 & 17 to identify the locations of fires.

For purposes of accuracy and limited comutational resources the neural network has been bypassed in the deployed version. Data of current fires are retrieved from the EOSDIS API. The current fires are then cross referenced with a list of all public schools in the United States. The US schools datasets is accessed from the Homeland Infrastructure Foundation-Level Data (HIFLD).

### Project Code
https://github.com/justinpaulturner/nasa_space_apps

### Data & Resources
https://registry.opendata.aws/noaa-goes/
