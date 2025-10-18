# RealEstatePRO 🏠

RealEstatePRO is an intelligent real estate search assistant that helps users find properties and analyze their surroundings using natural language queries. It combines vector search, geospatial analysis, and large language models to provide comprehensive property information.

## Features

- Natural language property search
- Price analysis and averages by location
- Find nearby amenities (restaurants, shops, transit stations, etc.)
- Calculate travel times between locations
- Interactive Streamlit web interface
- Vector similarity search for properties
- Distance and travel time calculations

## Tech Stack

### Language Models and APIs
- Google Gemini Pro (model: gemini-2.5-pro)
- Google Generative AI Embeddings (model: models/gemini-embedding-001)

### Main Libraries
- LangChain: Framework for building LLM applications
- FAISS: Vector similarity search
- Streamlit: Web interface
- Pandas: Data manipulation
- GeoPandas: Geospatial analysis
- Shapely: Geometric operations

### Data Storage
- FAISS vector stores for:
  - Property listings
  - Parks and recreational areas

## Tools and Capabilities

1. **Property Search (`retrieve`)**
   - Search properties using natural language
   - Returns detailed property information including:
     - Price, bedrooms, bathrooms
     - Square footage
     - Location (address, coordinates)

2. **Price Analysis (`average_price`)**
   - Calculate average property prices by location
   - Analyze market trends

3. **Places Near Location (`find_places_near_location`)**
   - Find amenities near properties
   - Search for specific types of places
   - Get location details and addresses

4. **Travel Time Calculator (`calculate_travel_time`)**
   - Calculate travel times between locations
   - Support for different transportation modes

6. **Places Near Location (`find_places_near_location`)**
   - Find amenities near properties
   - Search for specific types of places

## Project Structure

```
RealEstatePRO/
├── src/                  # Source code directory
│   ├── app.py           # Streamlit web application
│   ├── agent.py         # Custom LangChain agent implementation
│   └── tools.py         # Tool definitions and implementations
├── main.ipynb           # Development notebook and data processing
├── requirements.txt      # Project dependencies
└── faiss_index_dir/     # Property vector store
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd RealEstatePro
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**
   Create a `.env` file with:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   POSITION_API_KEY=your_position_api_key
   GOOGLE_APPLICATION_CREDENTIALS=path_to_your_credentials.json
   ```

4. **Run the Application**
   ```bash
   python main.py
   # or
   streamlit run src/app.py
   ```

## Using the Application

1. Enter your API keys in the sidebar
2. Type natural language queries in the chat interface, such as:
   - "Show me houses in New York with 3 bedrooms under $1,000,000"
   - "What's the average house price in Brooklyn?"
   - "Find parks near properties in Manhattan"
   - "Which houses are close to Central Park?"

## Development

The project uses a custom LangChain agent architecture with:
- Custom tool implementations
- Vector stores for efficient similarity search
- Geospatial calculations for location-based features
- Streaming responses for better user experience

### Data Processing

Property and park data are processed through:
1. Data cleaning and standardization
2. Embedding generation using Gemini
3. Vector store creation with FAISS
4. Geospatial data preparation for parks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- Support for more data sources
- Additional property analytics
- Enhanced geospatial features
- Market trend analysis
- Integration with more external APIs
- Mobile application support