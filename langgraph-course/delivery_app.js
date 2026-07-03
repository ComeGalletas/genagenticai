const vehicleSelect = document.getElementById('vehicle-select');
const demandSelect = document.getElementById('demand-select');
const locationSelect = document.getElementById('location-select');

const vehicleValue = document.getElementById('vehicle-value');
const weatherValue = document.getElementById('weather-value');
const demandValue = document.getElementById('demand-value');
const locationValue = document.getElementById('location-value');
const conditionValue = document.getElementById('condition-value');
const weatherApiValue = document.getElementById('weather-api-value');
const initialAmountInput = document.getElementById('initial-amount');
const estimatedValueEl = document.getElementById('estimated-value');

const capacityMap = {
  bike: 3,
  scooter: 5,
  car: 10,
  van: 20,
};

const weatherMultiplier = {
  sunny: 1.0,
  cloudy: 1.1,
  rainy: 1.9,
  windy: 1.3,
};

const demandMultiplier = {
  low: 0.9,
  medium: 1.0,
  high: 1.15,
  peak: 1.3,
};

const locationMultiplier = {
  hot: 1.05,
  rainy: 0.9,
  cold: 0.92,
  windy: 0.95,
  humid: 0.93,
};

const weatherApiLocations = {
  hot: 'Miami',
  rainy: 'Seattle',
  cold: 'Anchorage',
  windy: 'London',
  humid: 'Ho Chi Minh City',
};

const weatherApiCache = {};
const WEATHER_API_KEY = 'b8b199f8d1fc4d0d9ff202656230304';
const DEFAULT_LOCATION = 'Bogota';

async function fetchWeatherApi(locationType) {
  const city = weatherApiLocations[locationType] || DEFAULT_LOCATION;
  
  if (weatherApiCache[city]) {
    return weatherApiCache[city];
  }

  const url = `http://api.weatherapi.com/v1/current.json?key=${WEATHER_API_KEY}&q=${encodeURIComponent(city)}&aqi=no`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Weather API status ${response.status}`);
    }

    const data = await response.json();
    const currentWeather = {
      temperature: data.current?.temp_c || 0,
      windspeed: data.current?.wind_kph || 0,
      condition: data.current?.condition?.text || 'Unknown',
    };
    
    weatherApiCache[city] = currentWeather;
    return currentWeather;
  } catch (error) {
    console.warn('Weather API request failed:', error);
  }

  return null;
}

function updateSummary(weatherType) {
  vehicleValue.textContent = vehicleSelect.options[vehicleSelect.selectedIndex].text;
  weatherValue.textContent = weatherType || 'Loading...';
  demandValue.textContent = demandSelect.options[demandSelect.selectedIndex].text;
  locationValue.textContent = locationSelect.options[locationSelect.selectedIndex].text;
}

function calculateVehicleFactor(vehicleType) {
  return capacityMap[vehicleType] || 1;
}

function calculateEnvironmentFactor(weatherType, demandType, locationType) {
  const weatherFactor = weatherMultiplier[weatherType] || 1;
  const demandFactor = demandMultiplier[demandType] || 1;
  const locationFactor = locationMultiplier[locationType] || 1;
  return weatherFactor * demandFactor * locationFactor;
}

function calculateWeatherApiFactor(weatherData) {
  if (!weatherData) {
    return 1;
  }

  const temperature = weatherData.temperature || 0;
  const windspeed = weatherData.windspeed || 0;

  const temperatureFactor = temperature > 25 ? 1.05 : temperature < 5 ? 0.92 : 1.0;
  const windFactor = windspeed > 20 ? 0.95 : 1.0;

  return temperatureFactor * windFactor;
}

function mapTemperatureToWeatherType(temperature) {
  if (temperature === undefined || temperature === null) {
    return 'sunny';
  }
  if (temperature >= 28) {
    return 'sunny';
  }
  if (temperature >= 20) {
    return 'cloudy';
  }
  if (temperature >= 10) {
    return 'rainy';
  }
  return 'windy';
}

async function updateEstimate() {
  const initialAmount = Number(initialAmountInput.value) || 0;
  const vehicleType = vehicleSelect.value;
  const demandType = demandSelect.value;
  const locationType = locationSelect.value;

  const vehicleFactor = calculateVehicleFactor(vehicleType);
    const weatherData = await fetchWeatherApi(locationType);
    const weatherType = mapTemperatureToWeatherType(weatherData && weatherData.temperature);
  const environmentFactor = calculateEnvironmentFactor(weatherType, demandType, locationType);

  const apiWeatherFactor = calculateWeatherApiFactor(weatherData);
  const weatherFactor = weatherMultiplier[weatherType] || 1;

  if (weatherData) {
    weatherApiValue.textContent = `${weatherData.temperature}°C, wind ${weatherData.windspeed} km/h`;
    conditionValue.textContent = weatherData.condition || 'Unknown';
  } else {
    weatherApiValue.textContent = 'Unavailable';
    conditionValue.textContent = 'Unknown';
  }

  updateSummary(weatherType);

  const estimated = initialAmount * vehicleFactor * environmentFactor * weatherFactor * apiWeatherFactor;
  estimatedValueEl.textContent = `$${estimated.toFixed(2)}`;
}

async function updateAll() {
  updateSummary('Loading...');
  await updateEstimate();
}

vehicleSelect.addEventListener('change', updateAll);
demandSelect.addEventListener('change', updateAll);
locationSelect.addEventListener('change', updateAll);
initialAmountInput.addEventListener('input', updateEstimate);

updateAll();
