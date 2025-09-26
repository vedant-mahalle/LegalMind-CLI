
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS
import datetime
import os

def generate_pdf(template_vars, output_file='legal_notice.pdf'):
    """
    Generates a PDF from an HTML template.
    
    :param template_vars: A dictionary of variables to render in the template.
    :param output_file: The name of the output PDF file.
    """
    # Set up the Jinja2 environment to load templates from the current directory
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('template.html')

    # Render the HTML template with the provided data
    html_content = template.render(template_vars)

    # Load the CSS stylesheet
    css = CSS(filename='style.css')
    
    # Generate the PDF
    HTML(string=html_content).write_pdf(output_file, stylesheets=[css])
    print(f"✅ PDF generated successfully: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # --- Data extracted from the uploaded image ---
    # This dictionary contains all the dynamic data for the legal notice.
    notice_data = {
        # You can use local paths or base64 encoded strings for logos
        'logo_left_path': 'https://i.imgur.com/Y23PA55.png', # Placeholder for 'au fait' logo
        'logo_right_path': 'https://i.imgur.com/J3cW52D.png', # Placeholder for 'A' logo
        'recipient_name': 'Dr. Biswaroop Roy Chowdhury',
        'recipient_title': 'Chief Editor',
        'recipient_org': 'India Book of Records',
        'recipient_address_lines': [
            'B-121, Second Floor, Green Field Colony,',
            'Faridabad - 121010 (Haryana)'
        ],
        'date': '15th March 2021',
        'subject': 'Regarding false publication, Violations of Competition Rules & Defaming of individual or organizations.',
        'notice_points': [
            "That, our client Mr. Santosh Shukla (President, World Book of Records) Address : 18/3, Pardeshipura, Near Electronic Complex, Indore - 452 003 (Madhya Pradesh) India is a reputed and well-known personality of Indore and across the country, having an initiation of World Book of Records.",
            "That, the World Book of Records is a organization wherein our client deals with a programme with a motive to certify and encourage local and international talents, who does not get an opportunity of getting their talents acclaimed and certified. Under World Book of Records our client is continuously associated with many individuals, social, economical, financial and political organizations through whom our client gets to know the unacclaimed talents, may it be individuals or an organization.",
            "That, you as an organization are giving the records for felicitating and acclaiming the week or acts done by an individual or an organization as per your details presented on website you have been working since last many years for the same. Our client got an information this morning that you have published an information in form of an article on your official website under url http://indiabookofrecords.in/fakerecordbooks/ which was continuously circulated by you on various media platforms, showing that you as a record giving organization are the only one in India and neighbouring countries who is only working on the legal principle and guidelines laid down by the Government.",
            "That, in this regard under this article, you have given an INTERNATIONAL PROTOCOL for RECORDS (IPRs) under which their are five headings for identifying fake record books. As per the legal protocol and the information provided across the globe, their is no such International Protocol for Records which you have willfully placed on your official website which is truly forged."
        ]
    }
    
    # Call the function to generate the PDF
    generate_pdf(notice_data, output_file='Generated_Legal_Notice.pdf')
