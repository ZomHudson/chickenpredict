from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
from typing import Dict, List
import json
import os
import sys
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

app = Flask(__name__)
CORS(app)

# Malaysia timezone UTC+8
MY_TZ = timezone(timedelta(hours=8))

def malaysia_now() -> datetime:
    """Return current datetime in Malaysia timezone (UTC+8)"""
    return datetime.now(MY_TZ)

CSV_PATH = os.path.join(os.path.dirname(__file__), "ExFarmPrice.csv")

print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")


# ─── MongoDB Service ──────────────────────────────────────────────────────────

class MongoDBService:
    """Handles all MongoDB operations for recording predictions and stock data"""

    def __init__(self):
        self.client = None
        self.db = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Connect to MongoDB Atlas"""
        mongo_uri = os.getenv('MONGODB_URI', '')
        if not mongo_uri:
            print("WARNING: MONGODB_URI not set. Prediction recording disabled.")
            return

        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client['chicken_predictor']
            self.connected = True
            print("MongoDB connected successfully.")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection failed: {e}")
            self.connected = False

    def record_prediction(self, prediction: Dict) -> bool:
        """
        Save a full prediction snapshot to MongoDB.
        Called automatically every time /api/predict or /api/predict/week is hit.
        Returns True if saved successfully, False otherwise.
        """
        if not self.connected:
            return False

        try:
            stock = prediction.get('current_stock', {})
            price_info = prediction.get('price_info', {})
            calendar_event = prediction.get('calendar_event', {})
            factors = prediction.get('factors', {})

            # Parse raw float values from the formatted factor strings e.g. "+0.10 (+10.0%)"
            def parse_factor(factor_str: str) -> float:
                try:
                    return float(factor_str.split(' ')[0])
                except:
                    return 0.0

            document = {
                # When this record was saved (Malaysia time UTC+8)
                'recorded_at': malaysia_now(),

                # Prediction target
                'target_date': prediction.get('target_date'),

                # Demand output
                'demand_level': prediction.get('demand_level'),
                'demand_description': prediction.get('demand_description'),
                'recommendation': prediction.get('recommendation'),
                'overall_confidence': prediction.get('confidence'),

                # Stock snapshot at time of prediction
                'stock': {
                    'factory': stock.get('factory', 0),
                    'kiosk': stock.get('kiosk', 0),
                    'total': stock.get('total', 0),
                },

                # Price info
                'price': {
                    'value': price_info.get('price', 0.0),
                    'source': price_info.get('source', 'unknown'),
                    'confidence': price_info.get('confidence', 'unknown'),
                    'method': price_info.get('method', 'unknown'),
                    'forecast_factors': price_info.get('forecast_factors', {}),
                },

                # Calendar event at time of prediction
                'calendar_event': {
                    'has_event': calendar_event.get('has_event', False),
                    'event_name': calendar_event.get('event_name', 'Normal day'),
                    'event_type': calendar_event.get('type', 'normal'),
                    'factor': calendar_event.get('factor', 0.0),
                    'source': calendar_event.get('source', 'unknown'),
                },

                # Raw adjustment factors
                'factors': {
                    'inventory': parse_factor(factors.get('inventory_adjustment', '0')),
                    'price': parse_factor(factors.get('price_adjustment', '0')),
                    'calendar': parse_factor(factors.get('calendar_adjustment', '0')),
                    'day_of_week': parse_factor(factors.get('day_of_week_adjustment', '0')),
                    'total': parse_factor(factors.get('total_adjustment', '0')),
                },
            }

            self.db['predictions'].insert_one(document)
            print(f"Prediction recorded to MongoDB: {prediction.get('target_date')} → {prediction.get('demand_level')}")
            return True

        except Exception as e:
            print(f"Failed to record prediction: {e}")
            return False

    def get_prediction_history(self, days: int = 30, limit: int = 100) -> List[Dict]:
        """Retrieve past prediction records from MongoDB"""
        if not self.connected:
            return []

        try:
            cutoff = malaysia_now() - timedelta(days=days)
            cursor = (
                self.db['predictions']
                .find(
                    {'recorded_at': {'$gte': cutoff}},
                    {'_id': 0}   # exclude MongoDB internal _id field
                )
                .sort('recorded_at', DESCENDING)
                .limit(limit)
            )
            records = []
            for doc in cursor:
                # Convert datetime to ISO string for JSON serialisation
                if isinstance(doc.get('recorded_at'), datetime):
                    doc['recorded_at'] = doc['recorded_at'].isoformat()
                records.append(doc)
            return records

        except Exception as e:
            print(f"Failed to fetch prediction history: {e}")
            return []

    def get_stock_summary(self, days: int = 30) -> Dict:
        """Return average/min/max stock levels recorded over the last N days"""
        if not self.connected:
            return {}

        try:
            cutoff = malaysia_now() - timedelta(days=days)
            pipeline = [
                {'$match': {'recorded_at': {'$gte': cutoff}}},
                {'$group': {
                    '_id': None,
                    'avg_total_stock': {'$avg': '$stock.total'},
                    'min_total_stock': {'$min': '$stock.total'},
                    'max_total_stock': {'$max': '$stock.total'},
                    'avg_factory_stock': {'$avg': '$stock.factory'},
                    'avg_kiosk_stock': {'$avg': '$stock.kiosk'},
                    'total_records': {'$sum': 1},
                }}
            ]
            result = list(self.db['predictions'].aggregate(pipeline))
            if result:
                summary = result[0]
                summary.pop('_id', None)
                # Round all float values
                for key in summary:
                    if isinstance(summary[key], float):
                        summary[key] = round(summary[key], 2)
                return summary
            return {}

        except Exception as e:
            print(f"Failed to get stock summary: {e}")
            return {}

    def get_demand_distribution(self, days: int = 30) -> Dict:
        """Count how many times each demand level was predicted over N days"""
        if not self.connected:
            return {}

        try:
            cutoff = malaysia_now() - timedelta(days=days)
            pipeline = [
                {'$match': {'recorded_at': {'$gte': cutoff}}},
                {'$group': {
                    '_id': '$demand_level',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            result = list(self.db['predictions'].aggregate(pipeline))
            return {doc['_id']: doc['count'] for doc in result if doc['_id']}

        except Exception as e:
            print(f"Failed to get demand distribution: {e}")
            return {}


# Initialise MongoDB service (shared across all requests)
mongo_service = MongoDBService()


class LiveCalendarService:
    """Service to fetch Malaysian holidays from Calendarific API"""

    def __init__(self, api_key: str = None):
        self.calendarific_api_key = api_key
        self.cache = {}
        self.cache_expiry = {}

    def get_malaysian_holidays(self, year: int) -> List[Dict]:
        """Fetch Malaysian public holidays from Calendarific API"""
        cache_key = f"holidays_{year}"

        if cache_key in self.cache:
            if malaysia_now() < self.cache_expiry.get(cache_key, malaysia_now()):
                print(f"Using cached holidays for {year}")
                return self.cache[cache_key]

        if not self.calendarific_api_key:
            print("No Calendarific API key provided, cannot fetch holidays")
            return []

        try:
            url = "https://calendarific.com/api/v2/holidays"
            params = {
                'api_key': self.calendarific_api_key,
                'country': 'MY',
                'year': year,
                'type': 'national,local'
            }

            print(f"Fetching holidays from Calendarific for year {year}...")
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if response.status_code == 200 and data.get('meta', {}).get('code') == 200:
                holidays = []
                for holiday in data.get('response', {}).get('holidays', []):
                    holidays.append({
                        'name': holiday['name'],
                        'date': holiday['date']['iso'],
                        'type': holiday.get('type', ['national'])[0],
                        'description': holiday.get('description', '')
                    })

                self.cache[cache_key] = holidays
                self.cache_expiry[cache_key] = malaysia_now() + timedelta(hours=24)
                print(f"Successfully fetched {len(holidays)} holidays for {year}")
                return holidays
            else:
                print(f"Calendarific API error: {data}")
                return []

        except Exception as e:
            print(f"Error fetching holidays from API: {e}")
            return []

    def get_event_factor(self, holiday_name: str, holiday_type: str) -> Dict:
        """Determine demand factor based on holiday type and name"""

        major_festivals = {
            'Chinese New Year': {'factor': 0.40, 'pre_days': 5, 'post_days': 2},
            'Hari Raya Aidilfitri': {'factor': 0.50, 'pre_days': 5, 'post_days': 2},
            'Hari Raya Haji': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Hari Raya Aidiladha': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Christmas': {'factor': 0.30, 'pre_days': 3, 'post_days': 1},
            'Deepavali': {'factor': 0.25, 'pre_days': 3, 'post_days': 1},
            'Diwali': {'factor': 0.25, 'pre_days': 3, 'post_days': 1},
        }

        for festival_key, festival_data in major_festivals.items():
            if festival_key.lower() in holiday_name.lower():
                return festival_data

        if holiday_type == 'national':
            return {'factor': 0.20, 'pre_days': 2, 'post_days': 0}
        else:
            return {'factor': 0.15, 'pre_days': 1, 'post_days': 0}

    def get_calendar_events(self, target_date: datetime) -> Dict:
        """Get calendar events for a specific date from live API"""

        if not self.calendarific_api_key:
            print("No Calendarific API key configured, returning normal day")
            return {
                'has_event': False,
                'event_name': 'Normal day (no calendar data)',
                'factor': 0.0,
                'type': 'normal',
                'source': 'no_api_key'
            }

        try:
            current_year = malaysia_now().year
            target_year = target_date.year

            holidays = []
            if target_year == current_year:
                holidays.extend(self.get_malaysian_holidays(current_year))
            if target_year == current_year + 1:
                holidays.extend(self.get_malaysian_holidays(current_year + 1))

            if not holidays:
                print(f"No holiday data available for {target_date}")
                return self._get_rule_based_events(target_date)

            result = self._process_holidays(target_date, holidays, 'live_api')
            if result['has_event']:
                return result
            else:
                return self._get_rule_based_events(target_date)

        except Exception as e:
            print(f"Error in live calendar fetch: {e}")
            return self._get_rule_based_events(target_date)

    def _process_holidays(self, target_date: datetime, holidays: List[Dict], source: str) -> Dict:
        """Process holidays list and determine if target date has events"""

        target_date_str = target_date.strftime('%Y-%m-%d')

        for holiday in holidays:
            if holiday['date'] == target_date_str:
                event_config = self.get_event_factor(holiday['name'], holiday['type'])
                return {
                    'has_event': True,
                    'event_name': holiday['name'],
                    'factor': event_config['factor'],
                    'type': 'festival',
                    'source': source
                }

        for holiday in holidays:
            holiday_date = datetime.fromisoformat(holiday['date'])
            days_before = (holiday_date - target_date).days

            event_config = self.get_event_factor(holiday['name'], holiday['type'])

            if 0 < days_before <= event_config['pre_days']:
                proximity_factor = event_config['factor'] * (1 - days_before / event_config['pre_days']) * 0.7
                return {
                    'has_event': True,
                    'event_name': f"{days_before} days before {holiday['name']}",
                    'factor': proximity_factor,
                    'type': 'pre-festival',
                    'source': source
                }

            days_after = (target_date - holiday_date).days
            if 0 < days_after <= event_config['post_days']:
                return {
                    'has_event': True,
                    'event_name': f"{days_after} days after {holiday['name']}",
                    'factor': -0.25,
                    'type': 'post-festival',
                    'source': source
                }

        return {'has_event': False}

    def _get_rule_based_events(self, target_date: datetime) -> Dict:
        """Get rule-based events when no holiday data is available"""

        ramadan_info = self._check_ramadan_period(target_date)
        if ramadan_info['has_event']:
            return ramadan_info

        school_holiday_info = self._check_school_holidays(target_date)
        if school_holiday_info['has_event']:
            return school_holiday_info

        if target_date.weekday() == 4:
            return {
                'has_event': True,
                'event_name': 'Friday (weekend preparation)',
                'factor': 0.12,
                'type': 'friday',
                'source': 'rule_based'
            }

        return {
            'has_event': False,
            'event_name': 'Normal day',
            'factor': 0.0,
            'type': 'normal',
            'source': 'rule_based'
        }

    def _check_ramadan_period(self, target_date: datetime) -> Dict:
        """Check if date falls during Ramadan (approximate dates)"""
        ramadan_periods = {
            2025: {'start': datetime(2025, 3, 1), 'end': datetime(2025, 3, 30)},
            2026: {'start': datetime(2026, 2, 18), 'end': datetime(2026, 3, 19)},
            2027: {'start': datetime(2027, 2, 8), 'end': datetime(2027, 3, 9)},
        }

        year = target_date.year
        if year in ramadan_periods:
            period = ramadan_periods[year]
            if period['start'] <= target_date <= period['end']:
                days_to_end = (period['end'] - target_date).days
                if days_to_end <= 14:
                    ramadan_factor = 0.15 + (14 - days_to_end) / 14 * 0.20
                    return {
                        'has_event': True,
                        'event_name': 'Ramadan (approaching Raya)',
                        'factor': ramadan_factor,
                        'type': 'ramadan',
                        'source': 'calculated'
                    }
                else:
                    return {
                        'has_event': True,
                        'event_name': 'Ramadan',
                        'factor': 0.10,
                        'type': 'ramadan',
                        'source': 'calculated'
                    }

        return {'has_event': False}

    def _check_school_holidays(self, target_date: datetime) -> Dict:
        """Check approximate school holiday periods"""
        school_holidays = [
            {'start': datetime(2024, 11, 23), 'end': datetime(2025, 1, 5),  'name': 'Year End',   'factor': 0.15},
            {'start': datetime(2025, 3, 22),  'end': datetime(2025, 3, 30), 'name': 'Mid Year',   'factor': 0.12},
            {'start': datetime(2025, 5, 24),  'end': datetime(2025, 6, 8),  'name': 'Mid Year',   'factor': 0.12},
            {'start': datetime(2025, 8, 16),  'end': datetime(2025, 8, 24), 'name': 'Short Break','factor': 0.10},
            {'start': datetime(2025, 11, 22), 'end': datetime(2026, 1, 4),  'name': 'Year End',   'factor': 0.15},
            {'start': datetime(2026, 3, 28),  'end': datetime(2026, 4, 5),  'name': 'Mid Year',   'factor': 0.12},
            {'start': datetime(2026, 5, 23),  'end': datetime(2026, 6, 7),  'name': 'Mid Year',   'factor': 0.12},
            {'start': datetime(2026, 8, 22),  'end': datetime(2026, 8, 30), 'name': 'Short Break','factor': 0.10},
            {'start': datetime(2026, 11, 21), 'end': datetime(2027, 1, 3),  'name': 'Year End',   'factor': 0.15},
        ]

        for holiday in school_holidays:
            if holiday['start'] <= target_date <= holiday['end']:
                return {
                    'has_event': True,
                    'event_name': f"School Holiday ({holiday['name']})",
                    'factor': holiday['factor'],
                    'type': 'school-holiday',
                    'source': 'configured'
                }

        return {'has_event': False}


class ChickenRestockPredictor:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.RESTOCK_DAYS = [0, 3, 5]

        # Demand level thresholds based on total adjustment factor
        self.DEMAND_THRESHOLDS = {
            'low': -0.15,
            'medium_low': 0.0,
            'medium': 0.15,
            'medium_high': 0.30,
            'high': 0.30
        }

        calendarific_key = os.getenv('CALENDARIFIC_API_KEY', '')
        self.calendar_service = LiveCalendarService(calendarific_key)

    def fetch_current_stock(self) -> Dict:
        try:
            response = requests.get(self.api_url, timeout=10)
            data = response.json()

            if data.get('success', False):
                factory_stock = 0
                kiosk_stock = 0

                for item in data.get('factory_data', []):
                    if item['item_id'] == '11':
                        factory_stock = int(item['stock_count'])
                        break

                for kiosk in data.get('kiosk_data', []):
                    for item in kiosk.get('items', []):
                        if item['item_id'] == '11':
                            kiosk_stock += int(item['stock_count'])

                return {
                    'factory_stock': factory_stock,
                    'kiosk_stock': kiosk_stock
                }
            else:
                raise Exception("API returned unsuccessful response")

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return {'factory_stock': 500, 'kiosk_stock': 300}

    def parse_date_range(self, date_str):
        try:
            parts = date_str.split(' - ')
            if len(parts) == 2:
                end_date_str = parts[1].strip()
                return datetime.strptime(end_date_str, '%d.%m.%Y')
            return None
        except:
            return None

    def get_current_price(self) -> float:
        try:
            df = pd.read_csv(CSV_PATH)
            if len(df) == 0:
                return 6.5

            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            if len(df) == 0:
                return 6.5

            latest_row = df.sort_values('end_date', ascending=False).iloc[0]
            return float(latest_row['Avg_Price'])
        except Exception as e:
            print(f"Error reading price data: {e}")
            return 6.5

    def get_price_forecast(self, target_date: datetime) -> Dict:
        try:
            df = pd.read_csv(CSV_PATH)
            if len(df) == 0:
                return {
                    'forecasted_price': 6.5,
                    'confidence': 'Low',
                    'method': 'fallback',
                    'factors': {}
                }

            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            df = df.sort_values('end_date')

            if len(df) == 0:
                return {
                    'forecasted_price': 6.5,
                    'confidence': 'Low',
                    'method': 'fallback',
                    'factors': {}
                }

            current_price = float(df.iloc[-1]['Avg_Price'])
            days_ahead = (target_date - malaysia_now()).days

            # For dates within 3 days, use current price directly
            if days_ahead <= 3:
                return {
                    'forecasted_price': current_price,
                    'confidence': 'High',
                    'method': 'current',
                    'factors': {
                        'base_price': current_price,
                        'days_ahead': days_ahead,
                        'trend_adjustment': 0.0,
                        'festival_price_adjustment': 0.0,
                        'total_adjustment': 0.0
                    }
                }

            # ── Step 1: Calculate trend from recent CSV data ──────────────────
            weeks_back = min(8, len(df))
            recent_df = df.tail(weeks_back)

            trend_adjustment = 0.0
            if len(recent_df) >= 3:
                price_changes = recent_df['Avg_Price'].pct_change().dropna()
                avg_weekly_change = price_changes.mean()
                weeks_ahead = days_ahead / 7
                trend_adjustment = avg_weekly_change * weeks_ahead
                # Cap trend at ±15% — prices rarely move more than this
                trend_adjustment = max(-0.15, min(0.15, trend_adjustment))

            # ── Step 2: Festival price adjustment via Calendarific API ────────
            # Festivals push ex-farm prices up due to higher slaughter demand.
            # We use the same calendar service already used for demand,
            # so no hardcoded dates — works for any year automatically.
            festival_price_factor = 0.0
            festival_name_for_log = 'none'

            try:
                calendar_info = self.calendar_service.get_calendar_events(target_date)
                event_type = calendar_info.get('type', 'normal')
                event_factor = calendar_info.get('factor', 0.0)
                festival_name_for_log = calendar_info.get('event_name', 'none')

                if event_type == 'festival':
                    # On festival day itself — price spike at its peak
                    # Scale: 30% of the demand factor (price rises less than demand)
                    festival_price_factor = event_factor * 0.30

                elif event_type == 'pre-festival':
                    # Approaching festival — prices rise as slaughter increases
                    # Scale: 40% of the demand factor (pre-festival drives price harder)
                    festival_price_factor = event_factor * 0.40

                elif event_type == 'post-festival':
                    # After festival — oversupply causes slight price dip
                    festival_price_factor = -0.05

                elif event_type == 'ramadan':
                    # Ramadan period — steady elevated demand raises price moderately
                    festival_price_factor = 0.08

                elif event_type == 'school-holiday':
                    # School holidays — mild demand increase, minor price effect
                    festival_price_factor = 0.03

                elif event_type == 'friday':
                    # Friday prep — minimal price effect
                    festival_price_factor = 0.01

                # Cap festival price factor at ±20%
                festival_price_factor = max(-0.10, min(0.20, festival_price_factor))

            except Exception as cal_err:
                print(f"Calendar lookup failed for price forecast: {cal_err}")
                festival_price_factor = 0.0

            # ── Step 3: Combine trend + festival into final forecast ──────────
            total_adjustment = trend_adjustment + festival_price_factor
            # Overall price cap: allow up to +20% / -15% from base
            total_adjustment = max(-0.15, min(0.20, total_adjustment))

            forecasted_price = current_price * (1 + total_adjustment)
            # Absolute price bounds (RM5.00 – RM10.00)
            forecasted_price = max(5.0, min(10.0, forecasted_price))
            forecasted_price = round(forecasted_price, 2)

            # ── Step 4: Confidence — based on volatility + days ahead ─────────
            if len(recent_df) >= 3:
                price_volatility = recent_df['Avg_Price'].std() / recent_df['Avg_Price'].mean()
                if price_volatility < 0.05 and days_ahead <= 14:
                    confidence = 'High'
                elif price_volatility < 0.10 and days_ahead <= 21:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
            else:
                confidence = 'Low'

            return {
                'forecasted_price': forecasted_price,
                'confidence': confidence,
                'method': 'trend_calendar',
                'factors': {
                    'base_price': round(current_price, 2),
                    'trend_adjustment': round(trend_adjustment * 100, 2),
                    'festival_price_adjustment': round(festival_price_factor * 100, 2),
                    'total_adjustment': round(total_adjustment * 100, 2),
                    'festival_event': festival_name_for_log,
                    'days_ahead': days_ahead,
                    'weeks_of_data': weeks_back
                }
            }

        except Exception as e:
            print(f"Error forecasting price: {e}")
            return {
                'forecasted_price': 6.5,
                'confidence': 'Low',
                'method': 'fallback_error',
                'factors': {'error': str(e)}
            }

    def get_price_adjustment_factor(self, price: float) -> float:
        NORMAL_PRICE = 6.5
        HIGH_PRICE_THRESHOLD = 6.8
        LOW_PRICE_THRESHOLD = 6.2

        if price >= HIGH_PRICE_THRESHOLD:
            return 0.3
        elif price >= NORMAL_PRICE:
            return 0.15
        elif price <= LOW_PRICE_THRESHOLD:
            return -0.2
        else:
            return 0.0

    def get_calendar_events(self, target_date: datetime) -> Dict:
        return self.calendar_service.get_calendar_events(target_date)

    def calculate_inventory_factor(self, factory_stock: int, kiosk_stock: int) -> float:
        total_stock = factory_stock + kiosk_stock

        if total_stock < 100:
            return 0.5
        elif total_stock < 300:
            return 0.3
        elif total_stock < 500:
            return 0.1
        elif total_stock > 1500:
            return -0.3
        elif total_stock > 1000:
            return -0.15
        else:
            return 0.0

    def calculate_day_of_week_factor(self, target_date: datetime) -> float:
        weekday = target_date.weekday()

        if weekday == 5:
            return 0.15
        elif weekday == 3:
            return 0.05
        elif weekday == 0:
            return 0.0
        else:
            return 0.0

    def determine_demand_level(self, total_adjustment: float) -> Dict:
        if total_adjustment >= self.DEMAND_THRESHOLDS['high']:
            level = "High"
            description = "Significantly elevated demand expected"
            recommendation = "Prepare maximum stock. Consider ordering extra supplies."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium_high']:
            level = "Medium-High"
            description = "Above average demand anticipated"
            recommendation = "Stock above normal levels to meet increased demand."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium']:
            level = "Medium"
            description = "Normal to slightly elevated demand"
            recommendation = "Maintain standard stock levels with slight buffer."
        elif total_adjustment >= self.DEMAND_THRESHOLDS['medium_low']:
            level = "Medium-Low"
            description = "Normal to slightly below average demand"
            recommendation = "Standard stock levels appropriate."
        else:
            level = "Low"
            description = "Below average demand expected"
            recommendation = "Reduce stock levels to avoid excess inventory."

        return {
            'level': level,
            'description': description,
            'recommendation': recommendation,
            'adjustment_factor': round(total_adjustment, 3)
        }

    def predict_restock_demand(self, target_date: datetime = None) -> Dict:
        if target_date is None:
            today = malaysia_now()
            for i in range(7):
                check_date = today + timedelta(days=i)
                if check_date.weekday() in self.RESTOCK_DAYS:
                    target_date = check_date
                    break
            else:
                target_date = today + timedelta(days=1)

        stock_data = self.fetch_current_stock()

        # Always use forecasted price — works for any date including today/tomorrow
        # get_price_forecast() returns current price when days_ahead <= 3 anyway,
        # but now it also applies festival/trend adjustments even for near dates
        price_info = self.get_price_forecast(target_date)
        current_price = price_info['forecasted_price']
        price_source = price_info['method']  # 'current', 'trend_calendar', etc.

        inventory_factor = self.calculate_inventory_factor(
            stock_data['factory_stock'],
            stock_data['kiosk_stock']
        )

        price_factor = self.get_price_adjustment_factor(current_price)
        calendar_info = self.get_calendar_events(target_date)
        calendar_factor = calendar_info.get('factor', 0.0)
        day_factor = self.calculate_day_of_week_factor(target_date)

        total_adjustment = inventory_factor + price_factor + calendar_factor + day_factor
        demand_info = self.determine_demand_level(total_adjustment)

        result = {
            'target_date': target_date.strftime('%Y-%m-%d (%A)'),
            'demand_level': demand_info['level'],
            'demand_description': demand_info['description'],
            'recommendation': demand_info['recommendation'],
            'current_stock': {
                'factory': stock_data['factory_stock'],
                'kiosk': stock_data['kiosk_stock'],
                'total': stock_data['factory_stock'] + stock_data['kiosk_stock']
            },
            'factors': {
                'inventory_adjustment': f"{inventory_factor:+.2f} ({inventory_factor*100:+.1f}%)",
                'price_adjustment': f"{price_factor:+.2f} ({price_factor*100:+.1f}%)",
                'calendar_adjustment': f"{calendar_factor:+.2f} ({calendar_factor*100:+.1f}%)",
                'day_of_week_adjustment': f"{day_factor:+.2f} ({day_factor*100:+.1f}%)",
                'total_adjustment': f"{total_adjustment:+.2f} ({total_adjustment*100:+.1f}%)"
            },
            'calendar_event': calendar_info,
            'price_info': {
                'price': current_price,
                'source': price_source,
                'confidence': price_info['confidence'],
                'method': price_info['method'],
                'forecast_factors': price_info.get('factors', {})
            },
            'confidence': self._calculate_confidence(total_adjustment, price_info['confidence'])
        }

        # ── Auto-record every prediction to MongoDB ───────────────────────────
        mongo_service.record_prediction(result)

        return result

    def _calculate_confidence(self, total_adjustment: float, price_confidence: str) -> str:
        if abs(total_adjustment) > 0.5:
            quantity_confidence = "Medium"
        elif abs(total_adjustment) > 0.3:
            quantity_confidence = "Medium-High"
        else:
            quantity_confidence = "High"

        confidence_scores = {
            'High': 3,
            'Medium-High': 2,
            'Medium': 1,
            'Low': 0
        }

        qty_score = confidence_scores.get(quantity_confidence, 1)
        price_score = confidence_scores.get(price_confidence, 1)
        avg_score = (qty_score + price_score) / 2

        if avg_score >= 2.5:
            return "High"
        elif avg_score >= 1.5:
            return "Medium-High"
        elif avg_score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def predict_next_week(self) -> list:
        predictions = []
        today = malaysia_now()

        for i in range(14):
            check_date = today + timedelta(days=i)
            if check_date.weekday() in self.RESTOCK_DAYS:
                result = self.predict_restock_demand(check_date)
                predictions.append(result)

        return predictions

    def get_price_history(self, days: int = 90) -> Dict:
        try:
            df = pd.read_csv(CSV_PATH)
            if len(df) == 0:
                return {'success': False, 'error': 'No data available'}

            df['end_date'] = df['Date_Range'].apply(self.parse_date_range)
            df = df.dropna(subset=['end_date'])
            df = df.sort_values('end_date')

            cutoff_date = malaysia_now() - timedelta(days=days)
            recent_df = df[df['end_date'] > cutoff_date]

            history = []
            for _, row in recent_df.iterrows():
                history.append({
                    'date': row['end_date'].strftime('%Y-%m-%d'),
                    'price': float(row['Avg_Price'])
                })

            today = malaysia_now()
            for i in range(1, 15):
                forecast_date = today + timedelta(days=i)
                price_info = self.get_price_forecast(forecast_date)
                history.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'price': price_info['forecasted_price'],
                    'is_forecast': True,
                    'confidence': price_info['confidence']
                })

            return {
                'success': True,
                'data': history,
                'current_price': float(recent_df.iloc[-1]['Avg_Price']),
                'avg_price': float(recent_df['Avg_Price'].mean()),
                'min_price': float(recent_df['Avg_Price'].min()),
                'max_price': float(recent_df['Avg_Price'].max())
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}


# Initialize predictor
predictor = ChickenRestockPredictor(
    api_url="https://inventory.ayamgorengkimiez.my/api/analytics/96e27e560a23a5a21978005c3d69add802bfa5b9be3cb6c1f7735e51db80bfe2/overview"
)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        result = predictor.predict_restock_demand()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/week', methods=['GET'])
def get_weekly_predictions():
    try:
        predictions = predictor.predict_next_week()
        return jsonify({'success': True, 'data': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/price/current', methods=['GET'])
def get_current_price():
    try:
        price = predictor.get_current_price()
        return jsonify({'success': True, 'data': {'price': price, 'source': 'csv'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/price/forecast', methods=['GET'])
def get_price_forecast_route():
    try:
        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            target_date = malaysia_now() + timedelta(days=7)

        price_info = predictor.get_price_forecast(target_date)
        return jsonify({'success': True, 'data': price_info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/price/history', methods=['GET'])
def get_price_history():
    try:
        days = int(request.args.get('days', 90))
        result = predictor.get_price_history(days)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        prediction = predictor.predict_restock_demand()
        stock = prediction['current_stock']['total']
        alerts = []

        if stock < 300:
            alerts.append({
                'type': 'critical',
                'message': 'Critical stock level detected',
                'detail': f"Current stock ({stock}) is below minimum threshold of 300 units",
                'timestamp': malaysia_now().isoformat()
            })

        if stock < 500:
            alerts.append({
                'type': 'warning',
                'message': 'Low stock warning',
                'detail': f"Stock level at {stock} units. Consider restocking soon.",
                'timestamp': malaysia_now().isoformat()
            })

        if prediction['demand_level'] in ['High', 'Medium-High']:
            alerts.append({
                'type': 'info',
                'message': f"{prediction['demand_level']} demand period approaching",
                'detail': f"{prediction['recommendation']} Event: {prediction['calendar_event']['event_name']}",
                'timestamp': malaysia_now().isoformat()
            })

        price_info = prediction['price_info']
        if price_info['source'] == 'forecasted' and price_info['price'] > 7.0:
            alerts.append({
                'type': 'warning',
                'message': 'High price forecast',
                'detail': f"Ex-farm price forecasted at RM {price_info['price']:.2f} for {prediction['target_date']}",
                'timestamp': malaysia_now().isoformat()
            })

        return jsonify({'success': True, 'data': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calendar/holidays', methods=['GET'])
def get_holidays():
    try:
        year = int(request.args.get('year', malaysia_now().year))
        holidays = predictor.calendar_service.get_malaysian_holidays(year)

        return jsonify({
            'success': True,
            'year': year,
            'count': len(holidays),
            'holidays': holidays,
            'source': 'live_api' if holidays else 'no_data'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_prediction_history():
    """Get past prediction records from MongoDB"""
    try:
        days = int(request.args.get('days', 30))
        limit = int(request.args.get('limit', 100))

        if not mongo_service.connected:
            return jsonify({
                'success': False,
                'error': 'MongoDB not connected. Set MONGODB_URI environment variable.'
            }), 503

        records = mongo_service.get_prediction_history(days=days, limit=limit)
        return jsonify({
            'success': True,
            'count': len(records),
            'days': days,
            'data': records
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history/stats', methods=['GET'])
def get_history_stats():
    """Get stock summary + demand distribution from recorded predictions"""
    try:
        days = int(request.args.get('days', 30))

        if not mongo_service.connected:
            return jsonify({
                'success': False,
                'error': 'MongoDB not connected. Set MONGODB_URI environment variable.'
            }), 503

        stock_summary = mongo_service.get_stock_summary(days=days)
        demand_distribution = mongo_service.get_demand_distribution(days=days)

        return jsonify({
            'success': True,
            'days': days,
            'stock_summary': stock_summary,
            'demand_distribution': demand_distribution
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cron/record', methods=['GET'])
def cron_record():
    """Called automatically by Vercel Cron every day at 8AM — no user needed"""
    try:
        result = predictor.predict_restock_demand()
        return jsonify({
            'success': True,
            'message': 'Prediction recorded automatically by cron',
            'demand_level': result['demand_level'],
            'target_date': result['target_date'],
            'recorded_at': malaysia_now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': malaysia_now().isoformat(),
        'calendar_api_configured': bool(os.getenv('CALENDARIFIC_API_KEY')),
        'mongodb_connected': mongo_service.connected
    })


@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Chicken Restock Demand Predictor API',
        'status': 'running',
        'version': '2.3 - MongoDB Recording',
        'calendar_integration': 'live_api' if os.getenv('CALENDARIFIC_API_KEY') else 'rule_based_fallback',
        'mongodb_connected': mongo_service.connected,
        'demand_levels': ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
        'endpoints': [
            '/api/predict               - Get demand prediction (auto-recorded)',
            '/api/predict/week          - Get weekly predictions (auto-recorded)',
            '/api/price/current         - Current price',
            '/api/price/forecast        - Price forecast',
            '/api/price/history         - Price history',
            '/api/alerts                - Get alerts',
            '/api/calendar/holidays     - Get holidays',
            '/api/history               - Get recorded prediction history',
            '/api/history/stats         - Stock summary + demand distribution',
            '/api/cron/record           - Auto-record prediction (Vercel Cron)',
            '/health                    - Health check',
        ]
    })
