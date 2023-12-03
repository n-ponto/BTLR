import os
import asyncio
import python_weather
import datetime

UPDATE_INTERVAL = datetime.timedelta(minutes=10)


class Weather:

    def __init__(self):
        if os.name == 'nt':
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy())
        self.weather = None
        self.last_retrieved = None
        asyncio.run(self._update_weather())

    def get_weather(self):
        print('Getting weather...')
        if self.weather is None or self.last_retrieved is None or (datetime.datetime.now() - self.last_retrieved) > UPDATE_INTERVAL:
            asyncio.run(self._update_weather())
        return self._describe_weather()

    def _describe_weather(self):
        weather = self.weather
        todays_forecast = next(weather.forecasts)
        todays_high = todays_forecast.highest_temperature
        todays_low = todays_forecast.lowest_temperature
        snowfall = todays_forecast.snowfall
        
        description = f'Right now it\'s {weather.current.description}'
        description += f' and {weather.current.temperature}°,'
        description += f' with a low today of {todays_low}° and a high of {todays_high}°.'
        return description

    async def _update_weather(self):
        # declare the client. the measuring unit used defaults to the metric system (celcius, km/h, etc.)
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            # fetch a weather forecast from a city
            self.weather = await client.get('Seattle')
        self.last_retrieved = datetime.datetime.now()


if __name__ == '__main__':
    weather = Weather()
    print(weather.get_weather())
