### Predict damage grade caused by earthquake using building characteristics/location data.

#### Project description:
Following the 7.8 Mw Gorkha Earthquake on April 25, 2015, Nepal carried out a massive household survey using mobile technology to assess building damage in the earthquake-affected districts. Although the primary goal of this survey was to identify beneficiaries eligible for government assistance for housing reconstruction, it also collected other useful socio-economic information. In addition to housing reconstruction, this data serves a wide range of uses and users e.g. researchers, newly formed local governments, and citizens at large. Based on aspects of building location and construction, the goal is to predict the level of damage to buildings caused by the 2015 Gorkha earthquake in Nepal.

#### Data description
The data was collected through surveys by the Central Bureau of Statistics that work under the National Planning Commission Secretariat of Nepal. This survey is one of the largest post-disaster datasets ever collected, containing valuable information on earthquake impacts, household conditions, and socio-economic-demographic statistics.

The raw dataset contains 762106 datapoints and consists of 3 files:
1. Structure Data: Contains structural information of the properties
2. Damage Data: Contains Damage assesment data of the prroperties.
3. Ownership Data: Conntains geographical,legal data of the properties.

#### Data Source:
https://eq2015.npc.gov.np/#/

#### Data Size:
1. Number of Buildings: 700,000
2. Districts: 12
3. Wards: 945
4. Features: 40 ( Building constuction type/material, Building age, Height of building, # of floors etc, distance from earthquake epicenter

#### Modelling
LightGBM -  Train & Test - 69%
Gradient Boosting - Train - 66%, Test - 67%
Multinomial Logistics - Train - 65%, Test - 65%
Ordinal Logistics - Train - 64%, Test - 64%
Ensemble was the best selected model - Train 73.3%, Test - 69.3%

