cat > README.md <<'EOF'
# 🕷️ Spy_der — Real-Time SPY Options & Market Dashboard

A real-time **SPY options analytics** and **market monitoring dashboard** that integrates with the **Schwab API** to visualize live option chain data, volume, open interest, and volatility metrics.  
It also automatically sends updates to **Discord** during market hours.

<div align="center">
  <a href="https://github.com/jbass4642/Spy_der">
    <img src="https://img.shields.io/github/stars/jbass4642/Spy_der" alt="GitHub Repo stars"/>
  </a>
</div>

---

## ⚙️ Features

### 📊 Live Market & Options Data
- Real-time SPY option chain from Schwab API  
- Automatic updates every 2 minutes (during market hours)  
- Discord alerts with annotated charts  

### 🧠 Advanced Analytics
- **Put/Call Walls** — identifies major support and resistance  
- **POC (Point of Control)** — volume-weighted equilibrium level  
- **Delta, Vanna, and Charm** — visualize option Greek influence  
- **RSI Tracking** — detects overbought or oversold zones  
- **VIX Integration** — volatility confirmation signals  

### 🖥️ Dashboard Visuals
- Interactive Plotly charts  
- Heikin-Ashi candlesticks for SPY & VIX  
- Net volume, open interest, total exposure, and IV skew plots  
- Real-time options table view  

### 🔔 Discord Integration
- Automatic SPY updates every 2 minutes  
- Special alerts when price nears the Put Wall or VIX drops  
- Sends annotated chart images to your Discord server  

---

## 🧩 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jbass4642/Spy_der.git
   cd Spy_der
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   ```bash
   SCHWAB_APP_KEY=your_app_key_here
   SCHWAB_APP_SECRET=your_app_secret_here
   SCHWAB_CALLBACK_URL=https://127.0.0.1
   DISCORD_WEBHOOK=https://discord.com/api/webhooks/your_webhook_here
   ```

## Schwab API Setup

1. **Create a Schwab Developer Account**
   - Visit [Schwab Developer Portal](https://developer.schwab.com/)
   - Register for a developer account
   - Create a new application

2. **Get API Credentials**
   - App Key (Consumer Key)
   - App Secret (Consumer Secret)
   - Callback URL (for OAuth)

3. **Configure OAuth**
   - Set up the callback URL in your Schwab app settings
   - Ensure your callback URL matches the one in your `.env` file

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Access the dashboard**
   - Open your browser to `http://localhost:5001`
   - The application will start on port 5001 by default

3. **Using the Dashboard**
   - Enter a ticker symbol (e.g., SPY, SPX, AAPL)
   - Select expiration dates from the dropdown
   - Adjust strike range using the slider
   - Toggle different chart types and options
   - Enable/disable auto-update streaming

## Chart Types

### Price Chart
- Real-time candlestick or Heikin-Ashi charts
- Volume overlay
- Support for gamma and percentage GEX levels

### Exposure Charts
- **Gamma Exposure**: Shows market maker hedging requirements
- **Delta Exposure**: Directional exposure by strike
- **Vanna Exposure**: Volatility-price cross-sensitivity
- **Advanced Greeks**: Charm, Speed, and Vomma exposures

### Historical Bubble Levels
- Historical exposure tracking over the last hour
- Bubble charts showing exposure intensity over time
- Available for Gamma, Delta, and Vanna

### Volume Analysis
- Options volume by strike
- Call/Put volume ratios
- Premium analysis by strike

### Options Chain
- Sortable options chain table
- Real-time bid/ask/last prices
- Volume and open interest data
- Implied volatility display

## Configuration Options

### Strike Range
- Adjustable from 1% to 20% of current price
- Filters options within the specified range

### Chart Toggles
- Show/hide calls, puts, or net exposure
- Color intensity based on exposure values
- Multiple expiration date support

### Color Customization
- Customizable call and put colors
- Intensity-based color scaling

### Auto-Update
- Real-time streaming data updates
- Pause/resume functionality
- 1-second update intervals

## Database

The application uses SQLite to store historical bubble levels data:
- Automatic database initialization
- Stores minute-by-minute exposure data
- Automatic cleanup of old data

#

## License

This project is for educational and personal use only. Please comply with Schwab's API terms of service and any applicable regulations regarding financial data usage.

## Disclaimer

This software is for informational purposes only. It does not constitute financial advice. Trading options involves significant risk and may not be suitable for all investors. Always consult with a qualified financial advisor before making investment decisions.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review Schwab API documentation
- Ensure all dependencies are properly installed

## Contact
Josh Bass
GitHub: @jbass4642

Project: Spy_der