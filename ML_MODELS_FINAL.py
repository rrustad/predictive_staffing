# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql import Window
import pandas as pd
from datetime import datetime, date, timedelta
from pyspark.sql.types import (
    StructType,
    StructField,
    TimestampType,
    IntegerType,
    StringType,
)
import pytz
from pytz import timezone
import matplotlib.pyplot as plt
import seaborn as sns


class MLModelsBase:

    ED_LIST = ["CTZ ED", "CJN ED", "CLS ED"]
    LOCATION_ID_COLUMN = "assignedPatLocPointOfCare"
    UNIQUE_VISIT_ID_COLUMN = "visitID"
    ARRIVAL_TIMESTAMP_COLUMN = "admitDateTime"
    DEPARTURE_TIMESTAMP_COLUMN = "dischargeDateTime"
    TRAINING_START_DATE_OFFSET = 70
    TRAINING_END_DATE_OFFSET = 1

    def __init__(self, df) -> None:
        self.df = df

    def __repr__(self) -> str:
        return f"The hourly arrival model dataframe is: {self.df}\nThe location ID column is: {MLModelsBase.LOCATION_ID_COLUMN}\nThe unique visit ID column is: {MLModelsBase.UNIQUE_VISIT_ID_COLUMN}\nThe arrival timestamp column is: {MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN}\nThe departure timestamp column is: {MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN}"

    def clean_census_raw_data(self):
        """
        Create census raw data that is consumed by (1) Daily Arrival Totals Model (2) Within Day Hourly Arrivals Model (3) Length of Stay (LoS) Model


        Args:
            - df_raw: Input dataframe
            - training_start_date (str): Name of the column in df that contains start timestamp value
            - training_end_date_offset (int): How many days to skip from the current date


        Returns:
            - df: Dataframe with addtional columns top_of_the_hour and patient_time_spent.
                top_of_the_hour is top of the hour when the patient arrives and patient_time_spent is time spent by the patient in seconds every top of the hour
        """

        return (
            self.df.withColumn(
                "arrival_date_trunc",
                date_trunc("DD", col(MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN)),
            )
            .withColumn(
                "arrival_date_trunc_hourly",
                date_trunc("Hour", col(MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN)),
            )
            .withColumn("model_training_date", current_date())
            .where(col("assignedPatLocPointOfCare").isin(MLModelsBase.ED_LIST))
            .where(
                col("admitDateTime")
                > date_sub(
                    col("model_training_date"),
                    (
                        MLModelsBase.TRAINING_START_DATE_OFFSET
                        + MLModelsBase.TRAINING_END_DATE_OFFSET
                    ),
                )
            )
            .where(
                col("admitDateTime")
                < date_sub(
                    col("model_training_date"), MLModelsBase.TRAINING_END_DATE_OFFSET
                )
            )
            .select(
                "visitID",
                "assignedPatLocPointOfCare",
                "facility",
                "admitDateTime",
                "dischargeDateTime",
                "arrival_date_trunc",
                "arrival_date_trunc_hourly",
                "model_training_date",
            )
            .orderBy("assignedPatLocPointOfCare", "admitDateTime")
        )

    @staticmethod
    def create_forecast_df():

        schema = StructType(
            [
                StructField("forecast_timestamp", StringType(), True),
            ]
        )

        current_pacific_time = datetime.now(tz=pytz.utc).astimezone(
            timezone("US/Pacific")
        )
        dt = datetime(
            year=current_pacific_time.year,
            month=current_pacific_time.month,
            day=current_pacific_time.day,
            hour=current_pacific_time.hour,
            minute=0,
            second=0,
        )
        list_of_forecast_datetime = [
            (dt + timedelta(hours=num)) for num in range(7 * 24)
        ]
        list_of_datetime_tenor_tuples = []

        for top_of_the_hour in list_of_forecast_datetime:
            list_of_datetime_tenor_tuples.append((top_of_the_hour.isoformat(),))

        return (
            spark.createDataFrame(data=list_of_datetime_tenor_tuples, schema=schema)
            .withColumn("admitDateTime", to_timestamp(col("forecast_timestamp")))
            .withColumn("weekday", expr("dayofweek(admitDateTime)"))
            .withColumn("hourofday", expr("hour(admitDateTime)"))
            .drop("forecast_timestamp")
        )

    @staticmethod
    def explode_to_hourly(df, min_timestamp, max_timestamp):
        """
        Create hourly rows between two datetime values, min_timestamp, max_timestamp


        Args:
          - df: Input dataframe
          - min_timestamp (str): Name of the column in df that contains start timestamp value
          - max_timestamp (str): Name of the column in df that contains end timestamp value


        Returns:
          - df: Dataframe with addtional columns top_of_the_hour and patient_time_spent.
                top_of_the_hour is top of the hour when the patient arrives and patient_time_spent is time spent by the patient in seconds every top of the hour
        """

        return (
            df.withColumn("arrival_trunc", date_trunc("hour", col(min_timestamp)))
            .selectExpr(
                "*",
                f"sequence(arrival_trunc, {max_timestamp}, interval 1 hour) as dateArray",
            )
            .withColumn("arrival_truc_explode", explode(col("dateArray")))
            .withColumn(
                "arrival_diff",
                col(min_timestamp).cast("long")
                - col("arrival_truc_explode").cast("long"),
            )
            .withColumn(
                "departure_diff",
                col(max_timestamp).cast("long")
                - col("arrival_truc_explode").cast("long"),
            )
            .withColumn("top_of_the_hour", col("arrival_truc_explode"))
            .withColumn(
                "patient_time_spent",
                when(col("arrival_diff") > 0, lit(3600) - col("arrival_diff"))
                .when(col("departure_diff") < lit(3600), col("departure_diff"))
                .otherwise(lit(3600)),
            )
            .drop(
                "arrival_trunc",
                "arrival_truc_explode",
                "dateArray",
                "arrival_diff",
                "departure_diff",
            )
        )


class MLHourlyArrivals(MLModelsBase):
    def __init__(self, df, location):
        super().__init__(df)
        self.location = location

    def nhhp_estimator(self):
        """
        Create a dataframe that contains Non Homogeneous Possion Process(NHPP) model estimates


        Args:
            - None: None

        Returns:
            - df: Dataframe with 168 rows and three columns: (1) Day of the week (weekday) (2) Hour of the day (hourofday) and (3) Estimated poission intensity (hourly_arrival_rate)

        """

        hourly_arrivals = (
            self.clean_census_raw_data()
            .where(col("assignedPatLocPointOfCare") == lit(self.location))
            .groupBy("arrival_date_trunc_hourly")
            .agg(expr("count(visitID) as hourly_arrival_totals"))
            .withColumn("weekday", expr("dayofweek(arrival_date_trunc_hourly)"))
            .withColumn(
                "arrival_date_trunc", date_trunc("DD", col("arrival_date_trunc_hourly"))
            )
            .withColumn("week_of_year", weekofyear(col("arrival_date_trunc")))
            .select(
                "week_of_year",
                "arrival_date_trunc",
                "arrival_date_trunc_hourly",
                "weekday",
                "hourly_arrival_totals",
            )
            .withColumn("hourofday", expr("hour(arrival_date_trunc_hourly)"))
            .orderBy("week_of_year", "arrival_date_trunc_hourly", "weekday")
            .distinct()
        )

        hourly_window = (
            Window.partitionBy("week_of_year", "weekday")
            .orderBy("hourofday")
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
        daily_sum = sum(col("hourly_arrival_totals")).over(hourly_window)

        hourly_df_stage1 = hourly_arrivals.withColumn("daily_sum", daily_sum)

        hourly_arrival_model_final = (
            hourly_df_stage1.groupBy("weekday", "hourofday")
            .agg(
                expr("sum(hourly_arrival_totals) as hourly_arrival_sum"),
                expr("sum(daily_sum) as daily_sum"),
            )
            .withColumn("weekday", col("weekday"))
            .select("weekday", "hourofday", "hourly_arrival_sum", "daily_sum")
            .withColumn(
                "hourlyArrivalRate",
                round(col("hourly_arrival_sum") / col("daily_sum"), 4),
            )
            .orderBy("weekday", "hourofday")
            .drop("hourly_arrival_sum", "daily_sum")
            .distinct()
        )

        assert (
            hourly_arrival_model_final.count() == 168
        ), "The Non Homogeneous Possion Estimator(NHPP) must have 168 rows"

        return hourly_arrival_model_final

    def pick_best_model(self):
        return self.nhhp_estimator()


class MLLengthOfStay(MLModelsBase):
    def __init__(self, df, location):
        super().__init__(df)
        self.location = location

    def nhhp_estimator(self):
        los_model2_stage1 = (
            self.clean_census_raw_data()
            .where(col("assignedPatLocPointOfCare") == lit(self.location))
            .withColumn(
                "los",
                col("dischargeDateTime").cast("long")
                - col("admitDateTime").cast("long"),
            )
            .withColumn("weekday", expr("dayofweek(arrival_date_trunc_hourly)"))
            .withColumn("hourofday", expr("hour(arrival_date_trunc_hourly)"))
        )
        hourly_window = (
            Window.partitionBy("weekday", "hourofday")
            .orderBy("los")
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
        los_list = collect_list(col("los")).over(hourly_window)
        los_average = round(avg(col("los")).over(hourly_window))

        los_model2_stage2 = (
            los_model2_stage1.withColumn("los_list", los_list)
            .withColumn("los_list_size", size(col("los_list")))
            .withColumn("losAverage", los_average)
            .select("weekday", "hourofday", "losAverage")
            .distinct()
        )

        assert (
            los_model2_stage2.count() == 168
        ), "The Non Homogeneous Possion Estimator(NHPP) must have 168 rows"

        return los_model2_stage2

    def pick_best_model(self):
        return self.nhhp_estimator()


class MLDailyArrivals(MLModelsBase):
    def __init__(self, df, location):
        super().__init__(df)
        self.location = location

    def moving_average_estimator(self):
        daily_arrival_stage1 = (
            self.clean_census_raw_data()
            .where(col("assignedPatLocPointOfCare") == lit(self.location))
            .groupBy("arrival_date_trunc")
            .agg(expr("count(visitID) as daily_arrival_totals"))
            .withColumn("weekday", expr("dayofweek(arrival_date_trunc)"))
            .select("weekday", "arrival_date_trunc", "daily_arrival_totals")
            .orderBy("arrival_date_trunc", "weekday")
            .distinct()
        )
        daily_arrivals_model_final = (
            daily_arrival_stage1.groupBy("weekday")
            .agg(expr("round(avg(daily_arrival_totals)) as predDailyArrivals"))
            .select("weekday", "predDailyArrivals")
            .distinct()
        )

        assert (
            daily_arrivals_model_final.count() == 7
        ), "The daily arrival forecast must have 7 rows"

        return daily_arrivals_model_final

    def pick_best_model(self):
        return self.moving_average_estimator()


class GenerateForecast(MLModelsBase):
    def __init__(
        self,
        df,
        location: str,
        hourly_arrivals: MLHourlyArrivals,
        length_of_stay: MLLengthOfStay,
        daily_arrivals: MLDailyArrivals,
    ):
        super().__init__(df)
        self.location = location
        self.hourly_arrivals = hourly_arrivals
        self.length_of_stay = length_of_stay
        self.daily_arrivals = daily_arrivals

    def create_historical_df(self, history_length=7):
        historical_pred_df = (
            adtorca_adt_pv1_raw.where(
                col(MLModelsBase.LOCATION_ID_COLUMN) == lit(self.location)
            )
            .where(col(MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN).isNotNull())
            .withColumn("default_currenttime", current_timestamp())
            .withColumn(
                "current_datetime_pacific",
                from_utc_timestamp(col("default_currenttime"), "America/Los_Angeles"),
            )
            .withColumn(
                "current_date_trunc", date_trunc("DD", col("current_datetime_pacific"))
            )
            .withColumn(
                "hist_start_date", date_sub("current_date_trunc", history_length)
            )
            .where(col(MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN) >= col("hist_start_date"))
            .withColumn("forecast_stage", lit("historical"))
            .withColumn("weekday", expr("dayofweek(admitDateTime)"))
            .withColumn("hourofday", expr("hour(admitDateTime)"))
            .select(
                MLModelsBase.LOCATION_ID_COLUMN,
                "forecast_stage",
                MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN,
                MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN,
            )
            .orderBy(
                MLModelsBase.LOCATION_ID_COLUMN, MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN
            )
        )

        return historical_pred_df

    def create_current_df(self):
        current_df = (
            adtorca_adt_pv1_raw.where(
                col(MLModelsBase.LOCATION_ID_COLUMN) == lit(self.location)
            )
            .where(col(MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN).isNull())
            .withColumn("default_currenttime", current_timestamp())
            .withColumn(
                "current_datetime_pacific",
                from_utc_timestamp(col("default_currenttime"), "America/Los_Angeles"),
            )
            .withColumn(
                "current_date_trunc", date_trunc("DD", col("current_datetime_pacific"))
            )
            .withColumn("discharge_start_date", date_sub("current_date_trunc", 10))
            .where(
                col(MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN)
                >= col("discharge_start_date")
            )
            .withColumn("forecast_stage", lit("future"))
            .withColumn("weekday", expr("dayofweek(admitDateTime)"))
            .withColumn("hourofday", expr("hour(admitDateTime)"))
            .select(
                MLModelsBase.LOCATION_ID_COLUMN,
                "forecast_stage",
                MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN,
                MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN,
                "weekday",
                "hourofday",
            )
            .orderBy(
                MLModelsBase.LOCATION_ID_COLUMN, MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN
            )
        )

        return current_df

    def create_future_df(self):
        future_df = (
            MLModelsBase.create_forecast_df()
            .withColumn("forecast_stage", lit("future"))
            .withColumn(MLModelsBase.LOCATION_ID_COLUMN, lit(self.location))
            .withColumn(
                MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN, lit(None).cast(StringType())
            )
            .select(
                MLModelsBase.LOCATION_ID_COLUMN,
                "forecast_stage",
                MLModelsBase.ARRIVAL_TIMESTAMP_COLUMN,
                MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN,
                "weekday",
                "hourofday",
            )
            .orderBy(
                MLModelsBase.LOCATION_ID_COLUMN, MLModelsBase.DEPARTURE_TIMESTAMP_COLUMN
            )
        )

        return future_df

    def predicted_current_df(self):
        current_df = self.create_current_df()
        los_model2_final = self.length_of_stay.pick_best_model()
        join_list = ["weekday", "hourofday"]
        current_pred_df = (
            current_df.join(los_model2_final, join_list, "left")
            .withColumn(
                "dischargeDateTime",
                to_timestamp(col("admitDateTime").cast("long") + col("losAverage")),
            )
            .drop("losAverage", "los_median", "weekday", "hourofday")
        )

        return current_pred_df

    def predicted_future_df(self):
        join_list_totalarrivals = ["weekday"]
        join_list_hourlyarrivals = ["weekday", "hourofday"]
        daily_arrivals_model_final = self.daily_arrivals.pick_best_model()
        hourly_arrival_model_final = self.hourly_arrivals.pick_best_model()
        los_model2_final = self.length_of_stay.pick_best_model()
        future_df = self.create_future_df()

        future_pred_stage1_df = (
            future_df.join(daily_arrivals_model_final, join_list_totalarrivals, "left")
            .join(hourly_arrival_model_final, join_list_hourlyarrivals, "left")
            .withColumn(
                "pred_arrivals",
                round(col("predDailyArrivals") * col("hourlyArrivalRate")).cast("int"),
            )
            .drop("predDailyArrivals", "hourlyArrivalRate")
            .selectExpr("*", "sequence(1, pred_arrivals) as patientArray")
            .withColumn("patient_arrival_explode", explode(col("patientArray")))
            .drop("patientArray", "pred_arrivals", "patient_arrival_explode")
        )

        future_pred_df = (
            future_pred_stage1_df.join(
                los_model2_final, join_list_hourlyarrivals, "left"
            )
            .withColumn(
                "dischargeDateTime",
                to_timestamp(col("admitDateTime").cast("long") + col("losAverage")),
            )
            .drop("losAverage", "los_median", "weekday", "hourofday")
        )
        return future_pred_df

    def occupancy_level_df(self, history_length=7):
        historical_pred_df = self.create_historical_df()
        current_pred_df = self.predicted_current_df()
        future_pred_df = self.predicted_future_df()

        forecast_union = historical_pred_df.union(current_pred_df).union(future_pred_df)

        occupancy_hourly_df = MLModelsBase.explode_to_hourly(
            forecast_union, "admitDateTime", "dischargeDateTime"
        )

        hourly_occupancy_level = (
            occupancy_hourly_df.groupBy("assignedPatLocPointOfCare", "top_of_the_hour")
            .agg(
                expr("count(patient_time_spent) as hourlyOccupancy"),
                expr("first(forecast_stage) as forecastStage"),
            )
            .withColumn("default_currenttime", current_timestamp())
            .withColumn(
                "forecastPacificDateTime",
                from_utc_timestamp(col("default_currenttime"), "America/Los_Angeles"),
            )
            .withColumn(
                "hist_start_date", date_sub("forecastPacificDateTime", history_length)
            )
            .where(col("top_of_the_hour") >= to_date(col("hist_start_date")))
            .withColumn("hourlyDateTime", col("top_of_the_hour"))
            .select(
                "forecastPacificDateTime",
                "assignedPatLocPointOfCare",
                "hourlyDateTime",
                "forecastStage",
                "hourlyOccupancy",
            )
            .orderBy("assignedPatLocPointOfCare", "hourlyDateTime")
            .distinct()
        )

        return hourly_occupancy_level

    def occupancy_level_chart(self):
        hourly_occupancy_level = self.occupancy_level_df()
        hourly_occupancy_level_pandas = hourly_occupancy_level.toPandas()
        plt.figure(figsize=(20, 5))
        sns.relplot(
            x="hourlyDateTime",
            y="hourlyOccupancy",
            hue="forecastStage",
            kind="line",
            data=hourly_occupancy_level_pandas,
            height=5,
            aspect=4,
            linewidth=4,
        )


# COMMAND ----------

query = '''
select * from adtorca.adt_pv1
'''

adtorca_adt_pv1_raw = spark.sql(query)

# COMMAND ----------

test_ha = MLHourlyArrivals(location = 'CTZ ED' , df=adtorca_adt_pv1_raw)
test_los = MLLengthOfStay(location = 'CTZ ED' , df=adtorca_adt_pv1_raw)
test_da = MLDailyArrivals(location = 'CTZ ED' , df=adtorca_adt_pv1_raw)
test_pred = GenerateForecast(location = 'CTZ ED' , df=adtorca_adt_pv1_raw, hourly_arrivals = test_ha, length_of_stay=test_los, daily_arrivals=test_da)
test_pred()

# COMMAND ----------

test_pred.occupancy_level_chart()

# COMMAND ----------

#test_pred.occupancy_level_chart()

# COMMAND ----------

#test_pred.occupancy_level_chart()

# COMMAND ----------

#display(test_pred.occupancy_level_df())
