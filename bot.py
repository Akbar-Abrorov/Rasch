"""
Telegram Bot for Rasch Model Assessment Analysis
Handles calibration, ability estimation, and tier ranking
Requirements: python-telegram-bot==20.7
"""

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import io
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from datetime import datetime

# Telegram imports with error handling
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError as e:
    print("ERROR: Please install python-telegram-bot version 20.7")
    print("Run: pip install python-telegram-bot==20.7")
    raise e

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot Token - Replace with your actual token
BOT_TOKEN = "8578690075:AAFUHQQTPyM9tUlwtKxpV8go_YLPRGcMsog"


class RaschModel:
    """Rasch Model Implementation for Item Response Theory"""

    def __init__(self):
        self.item_difficulties = None
        self.person_abilities = None
        self.n_items = None
        self.n_persons = None

    def rasch_probability(self, ability, difficulty):
        """Calculate probability of correct response using Rasch model"""
        return np.exp(ability - difficulty) / (1 + np.exp(ability - difficulty))

    def log_likelihood(self, params, responses):
        """Calculate log-likelihood for optimization"""
        n_persons, n_items = responses.shape
        abilities = params[:n_persons]
        difficulties = params[n_persons:]

        ll = 0
        for i in range(n_persons):
            for j in range(n_items):
                if not np.isnan(responses[i, j]):
                    prob = self.rasch_probability(abilities[i], difficulties[j])
                    if responses[i, j] == 1:
                        ll += np.log(prob + 1e-10)
                    else:
                        ll += np.log(1 - prob + 1e-10)
        return -ll

    def calibrate(self, responses, max_iter=1000):
        """
        Calibrate Rasch model using Joint Maximum Likelihood Estimation
        responses: numpy array (n_persons x n_items) with 0/1 values
        """
        self.n_persons, self.n_items = responses.shape

        # Initialize parameters
        # Abilities: mean of person scores (normalized)
        person_scores = np.nanmean(responses, axis=1)
        init_abilities = np.log((person_scores + 0.01) / (1 - person_scores + 0.01))

        # Difficulties: mean of item scores (normalized)
        item_scores = np.nanmean(responses, axis=0)
        init_difficulties = -np.log((item_scores + 0.01) / (1 - item_scores + 0.01))

        # Combine parameters
        init_params = np.concatenate([init_abilities, init_difficulties])

        # Optimize
        result = minimize(
            self.log_likelihood,
            init_params,
            args=(responses,),
            method='BFGS',
            options={'maxiter': max_iter, 'disp': False}
        )

        # Extract results
        self.person_abilities = result.x[:self.n_persons]
        self.item_difficulties = result.x[self.n_persons:]

        # Center difficulties (constraint: mean difficulty = 0)
        mean_difficulty = np.mean(self.item_difficulties)
        self.item_difficulties -= mean_difficulty
        self.person_abilities -= mean_difficulty

        return self.person_abilities, self.item_difficulties

    def estimate_ability(self, response_pattern, item_difficulties):
        """Estimate ability for a single person given item difficulties"""
        def neg_likelihood(ability, pattern, difficulties):
            ll = 0
            for j, response in enumerate(pattern):
                if not np.isnan(response):
                    prob = self.rasch_probability(ability, difficulties[j])
                    if response == 1:
                        ll += np.log(prob + 1e-10)
                    else:
                        ll += np.log(1 - prob + 1e-10)
            return -ll

        # Initial guess based on raw score
        raw_score = np.nanmean(response_pattern)
        init_ability = np.log((raw_score + 0.01) / (1 - raw_score + 0.01))

        result = minimize(neg_likelihood, init_ability, args=(response_pattern, item_difficulties))
        return result.x[0]


class AssessmentProcessor:
    """Process assessment data and generate reports"""

    def __init__(self):
        self.rasch = RaschModel()
        self.correct_answers = None
        self.n_closed = 35
        self.n_open = 10
        self.total_questions = 45

    def parse_excel(self, file_bytes):
        """Parse Excel file with correct answers and student responses"""
        df = pd.read_excel(io.BytesIO(file_bytes))

        # First row should be correct answers
        self.correct_answers = df.iloc[0, 1:].values  # Skip first column (Student ID)

        # Remaining rows are student responses
        student_data = df.iloc[1:].copy()

        return student_data

    def score_responses(self, student_data):
        """Score student responses against correct answers"""
        scored_data = []

        for idx, row in student_data.iterrows():
            student_id = row.iloc[0]
            responses = row.iloc[1:].values

            scored_responses = []
            raw_score = 0

            for i, (student_ans, correct_ans) in enumerate(zip(responses, self.correct_answers)):
                if i < self.n_closed:
                    # Closed questions: exact match
                    score = 1 if str(student_ans).strip().upper() == str(correct_ans).strip().upper() else 0
                else:
                    # Open questions: numeric score (already provided)
                    try:
                        score = float(student_ans)
                        if score > 1:
                            score = 1  # Cap at 1
                    except:
                        score = 0

                scored_responses.append(score)
                raw_score += score

            scored_data.append({
                'student_id': student_id,
                'responses': scored_responses,
                'raw_score': raw_score
            })

        return scored_data

    def create_tiers(self, abilities):
        """Create tier rankings based on ability scores"""
        sorted_indices = np.argsort(abilities)[::-1]  # Descending order
        n_students = len(abilities)

        tiers = np.empty(n_students, dtype=object)

        # Define tier boundaries (you can adjust these)
        tier_names = ['S+ (Elite)', 'S (Excellent)', 'A+ (Very Good)', 'A (Good)',
                      'B+ (Above Average)', 'B (Average)', 'C+ (Below Average)',
                      'C (Needs Improvement)', 'D (Significant Gap)']

        percentiles = [0, 5, 15, 30, 50, 70, 85, 95, 100]

        for i, (start, end) in enumerate(zip(percentiles[:-1], percentiles[1:])):
            start_idx = int(n_students * start / 100)
            end_idx = int(n_students * end / 100)
            for idx in sorted_indices[start_idx:end_idx]:
                tiers[idx] = tier_names[i]

        return tiers

    def generate_report(self, scored_data, abilities, difficulties, tiers):
        """Generate comprehensive Excel report"""
        # Create results dataframe
        results = []
        for i, data in enumerate(scored_data):
            results.append({
                'Rank': np.where(np.argsort(abilities)[::-1] == i)[0][0] + 1,
                'Student ID': data['student_id'],
                'Raw Score': f"{data['raw_score']:.2f}/{self.total_questions}",
                'Percentage': f"{(data['raw_score']/self.total_questions)*100:.1f}%",
                'Ability (θ)': f"{abilities[i]:.3f}",
                'Tier': tiers[i]
            })

        # Sort by rank
        results_df = pd.DataFrame(results).sort_values('Rank')

        # Create item analysis dataframe
        item_analysis = []
        for i in range(self.total_questions):
            q_type = "Closed" if i < self.n_closed else "Open"
            responses = [d['responses'][i] for d in scored_data]

            item_analysis.append({
                'Question': f"Q{i+1}",
                'Type': q_type,
                'Difficulty (β)': f"{difficulties[i]:.3f}",
                'Correct Rate': f"{np.mean(responses)*100:.1f}%",
                'Discrimination': self._calculate_discrimination(responses, abilities)
            })

        item_df = pd.DataFrame(item_analysis)

        return results_df, item_df

    def _calculate_discrimination(self, item_responses, abilities):
        """Calculate item discrimination (point-biserial correlation)"""
        correlation = np.corrcoef(item_responses, abilities)[0, 1]
        if np.isnan(correlation):
            return "N/A"
        return f"{correlation:.3f}"

    def create_excel_output(self, results_df, item_df):
        """Create formatted Excel file"""
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write student results
            results_df.to_excel(writer, sheet_name='Student Results', index=False)

            # Write item analysis
            item_df.to_excel(writer, sheet_name='Item Analysis', index=False)

            # Format the sheets
            workbook = writer.book

            # Format Student Results sheet
            ws1 = workbook['Student Results']
            self._format_worksheet(ws1, results_df)

            # Format Item Analysis sheet
            ws2 = workbook['Item Analysis']
            self._format_worksheet(ws2, item_df)

            # Add summary sheet
            ws3 = workbook.create_sheet('Summary')
            self._create_summary_sheet(ws3, results_df, item_df)

        output.seek(0)
        return output

    def _format_worksheet(self, ws, df):
        """Apply formatting to worksheet"""
        # Header formatting
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # Adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width

        # Freeze header row
        ws.freeze_panes = 'A2'

    def _create_summary_sheet(self, ws, results_df, item_df):
        """Create summary statistics sheet"""
        summary_data = [
            ['ASSESSMENT SUMMARY REPORT', ''],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['', ''],
            ['STUDENT STATISTICS', ''],
            ['Total Students:', len(results_df)],
            ['Mean Raw Score:', f"{results_df['Raw Score'].str.split('/').str[0].astype(float).mean():.2f}"],
            ['Mean Ability (θ):', f"{results_df['Ability (θ)'].astype(float).mean():.3f}"],
            ['SD Ability (θ):', f"{results_df['Ability (θ)'].astype(float).std():.3f}"],
            ['', ''],
            ['ITEM STATISTICS', ''],
            ['Total Questions:', len(item_df)],
            ['Mean Difficulty (β):', f"{item_df['Difficulty (β)'].astype(float).mean():.3f}"],
            ['SD Difficulty (β):', f"{item_df['Difficulty (β)'].astype(float).std():.3f}"],
            ['Mean Correct Rate:', f"{item_df['Correct Rate'].str.rstrip('%').astype(float).mean():.1f}%"],
        ]

        for i, row_data in enumerate(summary_data, 1):
            ws.cell(row=i, column=1, value=row_data[0]).font = Font(bold=True)
            if len(row_data) > 1:
                ws.cell(row=i, column=2, value=row_data[1])

        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20


# Telegram Bot Handlers
processor = AssessmentProcessor()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    welcome_msg = (
        "🎓 *Welcome to Rasch Model Assessment Bot!*\n\n"
        "This bot analyzes student test results using the Rasch IRT model.\n\n"
        "*How to use:*\n"
        "1. Prepare an Excel file with:\n"
        "   - First row: Correct answers (Student ID, Q1, Q2, ..., Q45)\n"
        "   - Subsequent rows: Student responses\n"
        "   - Closed questions (1-35): A, B, C, D answers\n"
        "   - Open questions (36-45): Numeric scores (0-1)\n\n"
        "2. Send the Excel file to this bot\n"
        "3. Receive comprehensive analysis with:\n"
        "   - Student rankings and ability scores\n"
        "   - Tier classifications\n"
        "   - Item difficulty analysis\n\n"
        "*Commands:*\n"
        "/start - Show this message\n"
        "/help - Get detailed instructions\n\n"
        "Send your Excel file to begin! 📊"
    )
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help information"""
    help_msg = (
        "📋 *Excel File Format:*\n\n"
        "*Structure:*\n"
        "```\n"
        "Student ID | Q1  | Q2  | ... | Q45\n"
        "ANSWERS    | A   | B   |     | 0.8\n"
        "Student1   | A   | C   |     | 0.6\n"
        "Student2   | B   | B   |     | 1.0\n"
        "```\n\n"
        "*Important Notes:*\n"
        "- First row MUST be correct answers\n"
        "- Questions 1-35: Closed (A/B/C/D)\n"
        "- Questions 36-45: Open (0.0-1.0)\n"
        "- Student ID in first column\n\n"
        "*Output Includes:*\n"
        "✅ Student rankings and tiers\n"
        "✅ Rasch ability estimates (θ)\n"
        "✅ Item difficulty parameters (β)\n"
        "✅ Statistical summaries\n"
        "✅ Item discrimination indices\n\n"
        "Send your Excel file now! 🚀"
    )
    await update.message.reply_text(help_msg, parse_mode='Markdown')

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded Excel file"""
    try:
        # Notify processing started
        processing_msg = await update.message.reply_text(
            "⏳ Processing your file...\n"
            "This may take a moment for Rasch calibration."
        )

        # Download file
        file = await update.message.document.get_file()
        file_bytes = await file.download_as_bytearray()

        # Parse Excel
        student_data = processor.parse_excel(bytes(file_bytes))

        # Score responses
        scored_data = processor.score_responses(student_data)

        # Prepare response matrix for Rasch
        response_matrix = np.array([d['responses'] for d in scored_data])

        # Check if we have enough data
        if len(scored_data) < 10:
            await processing_msg.edit_text(
                "⚠️ Warning: Less than 10 students detected.\n"
                "Rasch calibration may be unstable.\n"
                "Recommended: 30+ students for reliable results.\n\n"
                "Proceeding with analysis..."
            )

        # Calibrate Rasch model
        abilities, difficulties = processor.rasch.calibrate(response_matrix)

        # Create tiers
        tiers = processor.create_tiers(abilities)

        # Generate reports
        results_df, item_df = processor.generate_report(
            scored_data, abilities, difficulties, tiers
        )

        # Create Excel output
        excel_output = processor.create_excel_output(results_df, item_df)

        # Send results
        await processing_msg.edit_text("✅ Analysis complete! Sending results...")

        # Send Excel file
        await update.message.reply_document(
            document=excel_output,
            filename=f"Rasch_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            caption=(
                f"📊 *Analysis Complete!*\n\n"
                f"👥 Students Analyzed: {len(scored_data)}\n"
                f"📝 Questions: {processor.total_questions}\n"
                f"📈 Mean Ability: {np.mean(abilities):.3f}\n"
                f"📉 Mean Difficulty: {np.mean(difficulties):.3f}\n\n"
                f"Check the Excel file for detailed results!"
            ),
            parse_mode='Markdown'
        )

        # Send top 5 students summary
        top_5 = results_df.head(5)
        summary = "*🏆 Top 5 Students:*\n\n"
        for _, row in top_5.iterrows():
            summary += f"{row['Rank']}. {row['Student ID']} - {row['Tier']}\n   Ability: {row['Ability (θ)']} | Score: {row['Percentage']}\n\n"

        await update.message.reply_text(summary, parse_mode='Markdown')

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ *Error processing file:*\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"Please check your Excel format and try again.\n"
            f"Use /help for format guidelines.",
            parse_mode='Markdown'
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    await update.message.reply_text(
        "Please send an Excel file (.xlsx) with your assessment data.\n"
        "Use /help to see the required format."
    )


def main():
    """Start the bot"""
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.Document.FileExtension("xlsx"), handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start bot
    print("🤖 Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()