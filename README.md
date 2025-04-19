# AI Research Navigator 🧠

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-00A67E?style=for-the-badge&color=00A67E)

A professional AI-powered research assistant that helps you discover and analyze the latest breakthroughs in artificial intelligence research (2023-2024).

## Features ✨

- **Latest Research Analysis**: Get comprehensive overviews of recent advancements in AI/ML
- **Key Paper Extraction**: Automatically identify and format the most significant publications
- **Trend Forecasting**: Predict future research directions and applications
- **Customizable Filters**: Focus on specific years (2022-2024) and research areas
- **Academic-Grade Outputs**: Properly formatted citations and structured analysis

## Technologies Used 🛠️

- **Groq API**: Ultra-fast LLM inference using LLaMA3-8B model
- **LangChain**: For prompt engineering and chained operations
- **Streamlit**: Beautiful, interactive web interface
- **Python 3.10+**: Backend logic and data processing

## Installation ⚙️

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ai-research-navigator.git
cd ai-research-navigator
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:

```bash
echo "groq_key=your_groq_api_key_here" > constants.py
```
## Usage 🚀 Run the Streamlit app:

``` bash
streamlit run app.py
```

# In the web interface:

- Select your research focus area
- Choose publication year filters
- Enter your research topic
- Click "Search" to generate analysis

# Explore the results:

- Research overview with key breakthroughs
- Formatted academic papers
- Future trend predictions

## Example Queries 🔍
- "Recent advances in transformer architectures"
- "Diffusion models for medical imaging 2024"
- "Few-shot learning applications in robotics"

# Configuration �

Customize the app by modifying:

1. app.py: Main application logic
2. constants.py: API keys and sensitive data
3. Prompt templates: Adjust for different output formats

## Contributing 🤝
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

LangChain team for the amazing orchestration framework
Streamlit for the intuitive web framework

[ Note: This tool provides AI-generated research summaries. Always verify critical information with primary sources.] 


This README includes:

1. **Professional badges** for key technologies
2. **Clear feature descriptions** with emoji visual cues
3. **Step-by-step installation** instructions
4. **Usage guide** with example queries
5. **Configuration notes** for customization
6. **Contribution guidelines**
7. **License information**
8. **Acknowledgments**

The formatting uses standard Markdown with:
- Consistent heading levels
- Code blocks for commands
- Clear section organization
- Professional tone throughout
