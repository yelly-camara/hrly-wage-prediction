from fastapi import FastAPI, Query
from enum import Enum
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from fastapi.responses import RedirectResponse

app = FastAPI(
    title="Canadian Hourly Wage Predictor",
    description="Trained on Statistics Canada Labour Force Survey Microdata Files",
    version="1.0"
)

class Industry(str, Enum):
    Agriculture = "Agriculture"
    Forestry = "Forestry and logging and support activities for forestry"
    Fishing = "Fishing, hunting and trapping"
    Mining = "Mining, quarrying, and oil and gas extraction"
    Utilities = "Utilities"
    Construction = "Construction"
    MfgDurable = "Manufacturing - durable goods"
    MfgNonDurable = "Manufacturing - non-durable goods"
    Wholesale = "Wholesale trade"
    Retail = "Retail trade"
    Transportation = "Transportation and warehousing"
    Finance = "Finance and insurance"
    RealEstate = "Real estate and rental and leasing"
    Professional = "Professional, scientific and technical services"
    BusinessSupport = "Business, building and other support services"
    Education = "Educational services"
    Health = "Health care and social assistance"
    Information = "Information, culture and recreation"
    Accommodation = "Accommodation and food services"
    Other = "Other services (except public administration)"
    Public = "Public administration"

INDUSTRY_CODE = {
    Industry.Agriculture: "01",
    Industry.Forestry: "02",
    Industry.Fishing: "03",
    Industry.Mining: "04",
    Industry.Utilities: "05",
    Industry.Construction: "06",
    Industry.MfgDurable: "07",
    Industry.MfgNonDurable: "08",
    Industry.Wholesale: "09",
    Industry.Retail: "10",
    Industry.Transportation: "11",
    Industry.Finance: "12",
    Industry.RealEstate: "13",
    Industry.Professional: "14",
    Industry.BusinessSupport: "15",
    Industry.Education: "16",
    Industry.Health: "17",
    Industry.Information: "18",
    Industry.Accommodation: "19",
    Industry.Other: "20",
    Industry.Public: "21",
}

class Occupation(str, Enum):
    SeniorManagement = "Legislative and senior management occupations"
    MiddleManagement = "Specialized middle management occupations"
    RetailManagement = "Middle management occupations in retail and wholesale trade and customer services"
    TradesManagement = "Middle management occupations in trades, transportation, production and utilities"
    FinanceProfessional = "Professional occupations in finance"
    BusinessProfessional = "Professional occupations in business"
    AdminSupervisors = "Administrative and financial supervisors and specialized administrative occupations"
    AdminSupport = "Administrative occupations and transportation logistics occupations"
    Logistics = "Administrative and financial support and supply chain logistics occupations"
    NaturalScience = "Professional occupations in natural sciences"
    AppliedScience = "Professional occupations in applied sciences (except engineering)"
    Engineering = "Professional occupations in engineering"
    TechScience = "Technical occupations related to natural and applied sciences"
    HealthProfessional = "Health treating and consultation services professionals"
    Therapy = "Therapy and assessment professionals"
    Nursing = "Nursing and allied health professionals"
    HealthTech = "Technical occupations in health"
    HealthSupport = "Assisting occupations in support of health services"
    Law = "Professional occupations in law"
    Education = "Professional occupations in education services"
    SocialServices = "Professional occupations in social and community services"
    Government = "Professional occupations in government services"
    PublicProtection = "Occupations in front-line public protection services"
    Paraprofessional = "Paraprofessional occupations in legal, social, community and education services"
    EducationSupport = "Assisting occupations in education and in legal and public protection"
    CareProviders = "Care providers and public protection support occupations and student monitors, crossing guards and related occupations"
    Arts = "Professional occupations in art and culture"
    TechArts = "Technical occupations in art, culture, sport"
    CultureSupport = "Occupations in art, culture and sport"
    SalesSupervisors = "Retail sales and service supervisors and specialized occupations in sales and services"
    Sales = "Occupations in sales and services"
    CustomerService = "Sales and service representatives and other customer and personal services occupations"
    SalesSupport = "Sales and service support occupations"
    TradesTech = "Technical trades and transportation officers and controllers"
    Trades = "General trades"
    TransportOps = "Mail and message distribution, other transport equipment operators and related maintenance workers"
    TransportLabour = "Helpers and labourers and other transport drivers, operators and labourers"
    ResourceSupervisors = "Supervisors and occupations in natural resources, agriculture and related production"
    ResourceWorkers = "Workers and labourers in natural resources, agriculture and related production"
    ProcessSupervisors = "Supervisors, central control and process operators in processing, manufacturing and utilities and aircraft assemblers and inspectors"
    MachineOperators = "Machine operators, assemblers and inspectors in processing, manufacturing and printing"
    Labourers = "Labourers in processing, manufacturing and utilities"


OCCUPATION_CODE = {
    Occupation.SeniorManagement: "01",
    Occupation.MiddleManagement: "02",
    Occupation.RetailManagement: "03",
    Occupation.TradesManagement: "04",
    Occupation.FinanceProfessional: "05",
    Occupation.BusinessProfessional: "06",
    Occupation.AdminSupervisors: "07",
    Occupation.AdminSupport: "08",
    Occupation.Logistics: "09",
    Occupation.NaturalScience: "10",
    Occupation.AppliedScience: "11",
    Occupation.Engineering: "12",
    Occupation.TechScience: "13",
    Occupation.HealthProfessional: "14",
    Occupation.Therapy: "15",
    Occupation.Nursing: "16",
    Occupation.HealthTech: "17",
    Occupation.HealthSupport: "18",
    Occupation.Law: "19",
    Occupation.Education: "20",
    Occupation.SocialServices: "21",
    Occupation.Government: "22",
    Occupation.PublicProtection: "23",
    Occupation.Paraprofessional: "24",
    Occupation.EducationSupport: "25",
    Occupation.CareProviders: "26",
    Occupation.Arts: "27",
    Occupation.TechArts: "28",
    Occupation.CultureSupport: "29",
    Occupation.SalesSupervisors: "30",
    Occupation.Sales: "31",
    Occupation.CustomerService: "32",
    Occupation.SalesSupport: "33",
    Occupation.TradesTech: "35",
    Occupation.Trades: "36",
    Occupation.TransportOps: "37",
    Occupation.TransportLabour: "38",
    Occupation.ResourceSupervisors: "39",
    Occupation.ResourceWorkers: "40",
    Occupation.ProcessSupervisors: "41",
    Occupation.MachineOperators: "42",
    Occupation.Labourers: "43",
}

class Education(str, Enum):
    NoSchooling = "0 to 8 years"
    SomeHighSchool = "Some high school"
    HighSchool = "High school graduate"
    SomePostSecondary = "Some postsecondary"
    Diploma = "Postsecondary certificate or diploma"
    Bachelors = "Bachelor's degree"
    Graduate = "Above bachelor's degree"

EDUCATION_CODE = {
    Education.NoSchooling: 0,
    Education.SomeHighSchool: 1,
    Education.HighSchool: 2,
    Education.SomePostSecondary: 3,
    Education.Diploma: 4,
    Education.Bachelors: 5,
    Education.Graduate: 6,
}

class EstablishmentSize(str, Enum):
    LessThan20 = "Less than 20 employees"
    From20To99 = "20 to 99 employees"
    From100To500 = "100 to 500 employees"
    MoreThan500 = "More than 500 employees"

ESTSIZE_CODE = {
    EstablishmentSize.LessThan20: "1",
    EstablishmentSize.From20To99: "2",
    EstablishmentSize.From100To500: "3",
    EstablishmentSize.MoreThan500: "4",
}

class Gender(str, Enum):
    Men = "Men+"
    Women = "Women+"

GENDER_CODE = {
    Gender.Men: "1",
    Gender.Women: "2",
}

class AgeGroup(str, Enum):
    Age_15_19 = "15 to 19 years"
    Age_20_24 = "20 to 24 years"
    Age_25_29 = "25 to 29 years"
    Age_30_34 = "30 to 34 years"
    Age_35_39 = "35 to 39 years"
    Age_40_44 = "40 to 44 years"
    Age_45_49 = "45 to 49 years"
    Age_50_54 = "50 to 54 years"
    Age_55_59 = "55 to 59 years"
    Age_60_64 = "60 to 64 years"
    Age_65_69 = "65 to 69 years"
    Age_70_plus = "70 and over"

AGE_CODE = {
    AgeGroup.Age_15_19: "01",
    AgeGroup.Age_20_24: "02",
    AgeGroup.Age_25_29: "03",
    AgeGroup.Age_30_34: "04",
    AgeGroup.Age_35_39: "05",
    AgeGroup.Age_40_44: "06",
    AgeGroup.Age_45_49: "07",
    AgeGroup.Age_50_54: "08",
    AgeGroup.Age_55_59: "09",
    AgeGroup.Age_60_64: "10",
    AgeGroup.Age_65_69: "11",
    AgeGroup.Age_70_plus: "12",
}

class Province(str, Enum):
    NewfoundlandAndLabrador = "Newfoundland and Labrador"
    PrinceEdwardIsland = "Prince Edward Island"
    NovaScotia = "Nova Scotia"
    NewBrunswick = "New Brunswick"
    Quebec = "Quebec"
    Ontario = "Ontario"
    Manitoba = "Manitoba"
    Saskatchewan = "Saskatchewan"
    Alberta = "Alberta"
    BritishColumbia = "British Columbia"
    
PROVINCE_CODE = {
    Province.NewfoundlandAndLabrador: "10",
    Province.PrinceEdwardIsland: "11",
    Province.NovaScotia: "12",
    Province.NewBrunswick: "13",
    Province.Quebec: "24",
    Province.Ontario: "35",
    Province.Manitoba: "46",
    Province.Saskatchewan: "47",
    Province.Alberta: "48",
    Province.BritishColumbia: "59",
}

class Union(str, Enum):
    Member = "Union member"
    CoveredByContract = "Not a member but covered by a union contract or collective agreement"
    NonUnionized = "Non-unionized"
    
UNION_CODE = {
    Union.Member: "1",
    Union.CoveredByContract: "2",
    Union.NonUnionized: "3",
}

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
model_path = os.path.join(base_dir, "models", "model_v1.cbm")
model = CatBoostRegressor() 
model.load_model(model_path) 

@app.get("/")
def get_info():
    return {
        "app_name": "Hourly Wage Predictor",
        "version": "1.0.0",
        "model_type": "CatBoost Regressor",
        "author": "Yelly Camara"
    }
    
@app.get("/predict", description="""
    **Model Disclosure:** This model predicts hourly wages based on historical trends. 
    Our analysis indicates higher variance in predictions for men and highly educated professionals.
""")
@app.get("/predict")
def predict_one(
    occupation: Occupation = Query(
        Occupation.SeniorManagement, 
        description="Current professional role or job category"
    ),
    industry: Industry = Query(
        Industry.Agriculture, 
        description="General industry or business sector"
    ),
    education: Education = Query(
        Education.NoSchooling, 
        description="Highest level of formal schooling completed"
    ),
    tenure: int = Query(
        1, ge=0, le=240, 
        description="Employment duration in months"
    ),
    establishmentsize: EstablishmentSize = Query(
        EstablishmentSize.LessThan20, 
        description="Workforce size of the establishment"
    ),
    gender: Gender = Query(
        Gender.Men, 
        description="Gender identity"
    ),
    age: AgeGroup = Query(
        AgeGroup.Age_15_19, 
        description="Current range"
    ),
    province: Province = Query(
        Province.NewfoundlandAndLabrador, 
        description="Province of employment"
    ),
    union: Union = Query(
        Union.Member, 
        description="Union membership status"
    ),
    usualhours: float = Query(
        24, ge=0.1, le=99, 
        description="Total weekly hours"
    )
):


    input_data = {
        "NOC_43": OCCUPATION_CODE[occupation.value],
        "NAICS_21": INDUSTRY_CODE[industry.value],
        "EDUC": EDUCATION_CODE[education.value],
        "TENURE": tenure,
        "ESTSIZE": ESTSIZE_CODE[establishmentsize.value],
        "GENDER": GENDER_CODE[gender.value],
        "AGE_12": AGE_CODE[age.value],
        "PROV": PROVINCE_CODE[province.value],
        "UNION": UNION_CODE[union.value],
        "UHRSMAIN": usualhours * 10
    }
    

    result = model.predict(pd.DataFrame([input_data]))
    prediction = result/100 


    return {
        "Predicted Hourly Wage": f"${prediction[0]:.0f}"
    }