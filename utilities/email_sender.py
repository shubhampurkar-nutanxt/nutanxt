import mimetypes
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
from email.mime.application import MIMEApplication
import time
import datetime
import pandas as pd
import numpy as np
import json
import io
import os
from config import ER_BCC,DEFAULT_EMAIL,CALIBRATION_EMAIL_CC,ER_BCC_Test,DEFAULT_EMAIL_Test,CALIBRATION_EMAIL_CC_Test,DIR_CONFIG,expert_sender,receiver_mail_chat,expert_NameServer,expert_passcode,host,port,receiver_mail_chat,support_sender_mail,support_passcode,support_NameServer

def get_customer():
    try:
        with open(os.path.join(DIR_CONFIG, 'app_settings.json')) as f:
            settings = json.load(f)
        customer = settings.get('manager_email', '').strip()
        if "nutanxt" in customer:
            to = DEFAULT_EMAIL_Test
            ER_bcc = ER_BCC_Test
            calibration_bcc = CALIBRATION_EMAIL_CC_Test
        else:
            to = DEFAULT_EMAIL
            ER_bcc = ER_BCC
            calibration_bcc = CALIBRATION_EMAIL_CC
        return {"to": to,"ER_bcc":ER_bcc,"calibration_bcc":calibration_bcc}
    except FileNotFoundError:
        print("Config file not found.")
        return {"to": DEFAULT_EMAIL,"ER_bcc":ER_BCC,"calibration_bcc":CALIBRATION_EMAIL_CC}
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return {"to": DEFAULT_EMAIL,"ER_bcc":ER_BCC,"calibration_bcc":CALIBRATION_EMAIL_CC}

class Email_Sender:
    def __init__(self,cred):
        self.__cred = cred
        self.__smtp_server = 0
     
    @classmethod
    def create_newsession(cls,cred):
        return cls(cred)
    
    
    def __create_session(self):
        try:
            if self.__cred['port'] == 587:
                try:
                    self.__smtp_server = smtplib.SMTP(self.__cred['host'], self.__cred['port'])
                    self.__smtp_server.ehlo()
                    self.__smtp_server.starttls()
                    self.__smtp_server.ehlo()
                    self.__smtp_server.login(self.__cred['sender_mail'], self.__cred['passcode'])
                    print("Connected and authenticated to SMTP server on port 587.")
                except smtplib.SMTPException as e:
                    print(f"Failed to connect or authenticate with the SMTP server on port 587: {e}")
                    self.__smtp_server.quit()
                    raise
            elif self.__cred['port'] == 465:
                try:
                    self.__smtp_server = smtplib.SMTP_SSL(self.__cred['host'], self.__cred['port'])
                    self.__smtp_server.login(self.__cred['sender_mail'], self.__cred['passcode'])
                    print("Connected and authenticated to SMTP server on port 465.")
                except smtplib.SMTPException as e:
                    print(f"Failed to connect or authenticate with the SMTP server on port 465: {e}")
                    self.__smtp_server.quit()
                    raise
            else:
                raise ValueError(f"Unsupported port: {self.__cred['port']}")

        except Exception as e:
                print(f"Error in Email module - {e}")
                self.__smtp_server = 0
    
    @staticmethod
    def check_session(server):
        try:
            res = server.noop()[0]
        except:
            res = -1
        return True if res==250 else False
    
    
    def __create_EmailBody(self,content):
        body = EmailMessage()
        body["Subject"] = content.get("subject","NO Subject")
        body["From"] = "Bitbucket_Report "+ self.__cred['sender_mail']
        body["To"] = self.__cred['receiver_mail']
        body['X-Priority'] = '1'
        body['X-MSMail-Priority'] = 'High'
        body['Importance'] = 'High'
        
        html_msg = content.get("html","<body>Please find the attachment</body>")
        
        body.add_alternative(html_msg, subtype='html')
        
        try:
            report_id = "Oak_Ds_Test_Report.html"
            with open(content.get("report_path",'/opt/atlassian/pipelines/agent/build/test-reports/report.html')) as fp:
                report_data = fp.read()
            body.add_attachment(report_data, subtype='html', filename=report_id)
        except: print("Test Report html is not found")
        try:
            report_id = "Oak_Ds_Function_map.html"
            with open(content.get("function_report",'/opt/atlassian/pipelines/agent/build/test-reports/Function_Map.html')) as fp:
                report_data = fp.read()
            body.add_attachment(report_data, subtype='html', filename=report_id)
        except: print("Function map html is not found")
        return body

    def __followup_support(self,content):
        print("In followup_support")
        body = EmailMessage()
        body["Subject"] = content.get("subject","NO Subject")
        body["From"] = self.__cred['sender_mail']
        body["To"] = self.__cred['receiver_mail']
        body['X-Priority'] = '1'
        body['X-MSMail-Priority'] = 'High'
        body['Importance'] = 'High'
        
        html_msg = content.get("html","<body>Please find the attachment</body>")
        
        body.add_alternative(html_msg, subtype='html')
    
        return body
     
    
    def __create_supportEmailBody(self,content):
        
        sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
        
        body = EmailMessage()
        body['Subject'] = f'Oak Analytics - NR-{content.get("device")} Requesting for Expert Review on scan_id {content.get("scan_id")}'
        body['From'] = f'''{self.__cred["sender_mail"]}'''
        body['To'] =  self.__cred['receiver_mail']
        body['Bcc'] = self.__cred['bcc']
        
        html_msg = f""" <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href='https://fonts.googleapis.com/css?family=Nunito:400,300' rel='stylesheet' type='text/css'>
           
          </head>
          
        <body><p>
        Hello, <br/><br/>
        
        The NR_{content.get("device")} user with id <span style={sender_email_style}>{content.get("user_id")}</span>, requested for Expert Review.<br/>
        <h4>Details</h4>
        <ul>
        <li>Scan Id: <span style='font-weight: bold; color: green;'>{content.get("scan_id")}</span></li>
        <li>User Details: <span style='font-weight: bold; color: blue;'>{content.get("user_id")}</span></li>
        </ul>
            
        
        <br/>Please find the attached scan file, hope this helps for analysis
        <br/>
        <br/>
        Thanks, and Regards..
        <br/><br/><br/>
        <span style='font-weight: bold,color: grey,font-size: 10px'>Please Note: Dont reply for this mail, it is a automaticed mail</span>
        </p></body></html>"""
        
        body.add_alternative(html_msg, subtype='html')
        
        json_file_path = content.get("filepath",None)
        xcal_data = None
        if json_file_path is not None:
            json_content = open(json_file_path, 'rb').read()
            xcal_data = json.loads(json_content.decode('utf-8')).get('xcal',None)
            if xcal_data is not None: xcal_data = eval(xcal_data)

            attachment = MIMEApplication(json_content)
            attachment.add_header('Content-Disposition', f'attachment; filename="{content.get("scan_id")}.json"')
            body.attach(attachment)

        if xcal_data is not None:
            csv_content = pd.DataFrame({"X": list(np.arange(200, 2601)), 'Y': xcal_data}).set_index("X").T.to_csv()
            csv_content_bytes = csv_content.encode() 
            csv_attachment = MIMEApplication(csv_content_bytes) 
            csv_attachment.add_header('Content-Disposition', f'attachment; filename="{content.get("scan_id")}.csv"')
            body.attach(csv_attachment)
            
        return body
            
    
    # def _create_id_sync_email_body(self,content):
    #     sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
    #     scan_list = content.get("scan_list")
    #     print("scan_list",scan_list)
    #     device_id = content.get('device')
    #     body = EmailMessage()

    #     if len(scan_list) == 1:
    #         body['Subject'] = f"Oak Analytics - NR-{device_id} Scan Sync Request [Scan ID: {scan_list[0]}]"
    #         scan_id = f"{scan_list[0]}"
    #     else:
    #         body['Subject'] = f"Oak Analytics - NR-{device_id} Batch Scan Sync Request [Scan IDs: {scan_list[0]} to {scan_list[-1]}]"
    #         scan_id = f"{scan_list[0]}"
    #     body['From'] = f'''{self.__cred["sender_mail"]}'''
    #     body['To'] =  self.__cred['receiver_mail']
    #     body['Cc'] = self.__cred['cc']
        
    #     html_msg = f""" 
    #         <html>
    #             <body>
    #                 <p>Hello Team,<br><br>
    #                 This is an automated scan sync request from device <strong>NR-{device_id}</strong>.<br><br>
    #                 <strong>Scan Details:</strong>
    #                 <ul>
    #                 <li>Scan Count: {len(scan_list)}</li>
    #                 </ul>
    #                 <strong>Scan IDs:</strong>
    #                 <ul>
    #                 {scan_list}
    #                 </ul>
    #                 Attached JSON contains the scan data for syncing to AWS.<br><br>
    #                 Regards,<br>
    #                 Oak Analytics System<br><br>
    #                 <span style="font-size: 10px; color: grey;">Note: This is an automated message. Please do not reply.</span>
    #                 </p>
    #             </body>
    #         </html>        
    #         """
        
    #     body.add_alternative(html_msg, subtype='html')
    #     json_file_path = content.get("filepath",None)
    #     xcal_data = None
    #     if json_file_path and os.path.exists(json_file_path):
    #         with open(json_file_path, 'rb') as f:
    #             json_content = f.read()
    #         attachment = MIMEApplication(json_content)
    #         attachment.add_header('Content-Disposition', f'attachment; filename="{scan_id}".json"')
    #         body.attach(attachment)
    #     return body   
    def _create_id_sync_email_body(self, content):
        sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
        scan_list = content.get("scan_list") or []  # Ensures scan_list is a list
        print("scan_list", scan_list)
        device_id = content.get('device', 'Unknown')
        body = EmailMessage()

        if len(scan_list) == 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} Scan Sync Request [Scan ID: {scan_list[0]}]"
            scan_id = scan_list[0]
        elif len(scan_list) > 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} All Scan Sync Request [Scan IDs: {scan_list[0]} to {scan_list[-1]}]"
            scan_id = scan_list[0]
        else:
            body['Subject'] = f"Oak Analytics - NR-{device_id} Scan Sync Request [No Scan ID]"
            scan_id = "scan_data"

        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        # body['Cc'] = self.__cred['cc']

        scan_id_list_html = ''.join(f"<li>{s}</li>" for s in scan_list) if scan_list else "<li>No scans available</li>"

        html_msg = f""" 
            <html>
                <body>
                    <p>Hello Team,<br><br>
                    This is an automated scan sync request from device <strong>NR-{device_id}</strong>.<br><br>
                    <strong>Scan Details:</strong>
                    <ul>
                        <li>Scan Count: {len(scan_list)}</li>
                    </ul>
                    <strong>Scan IDs:</strong>
                    <ul>
                        {scan_id_list_html}
                    </ul>
                    Attached JSON contains the scan data for syncing to AWS.<br><br>
                    Regards,<br>
                    Oak Analytics System<br><br>
                    <span style="font-size: 10px; color: grey;">Note: This is an automated message. Please do not reply.</span>
                    </p>
                </body>
            </html>        
            """

        body.add_alternative(html_msg, subtype='html')

        json_file_path = content.get("filepath")
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'rb') as f:
                json_content = f.read()
            attachment = MIMEApplication(json_content)
            attachment.add_header('Content-Disposition', f'attachment; filename="{scan_id}.json"')
            body.attach(attachment)

        return body
    def _create_callibration_data_sync_email_body(self, content):
        sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
        scan_list = content.get("scan_list") or []  # Ensures scan_list is a list
        print("scan_list", scan_list)
        device_id = content.get('device', 'Unknown')
        body = EmailMessage()

        if len(scan_list) == 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} x-calibration Scan Sync Request [Scan ID: {scan_list[0]}]"
            scan_id = scan_list[0]
        elif len(scan_list) > 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} All x-calibration Scan Sync Request [Scan IDs: {scan_list[0]} to {scan_list[-1]}]"
            scan_id = scan_list[0]
        else:
            body['Subject'] = f"Oak Analytics - NR-{device_id} x-calibration Scan Sync Request [No Scan ID]"
            scan_id = "scan_data"

        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        # body['Cc'] = self.__cred['cc']

        scan_id_list_html = ''.join(f"<li>{s}</li>" for s in scan_list) if scan_list else "<li>No scans available</li>"

        html_msg = f""" 
            <html>
                <body>
                    <p>Hello Team,<br><br>
                    This is an automated x-calibration scan sync request from device <strong>NR-{device_id}</strong>.<br><br>
                    <strong>Scan Details:</strong>
                    <ul>
                        <li>Scan Count: {len(scan_list)}</li>
                    </ul>
                    <strong>Scan IDs:</strong>
                    <ul>
                        {scan_id_list_html}
                    </ul>
                    Attached JSON contains the x-calibration scan data for syncing to AWS.<br><br>
                    Regards,<br>
                    Oak Analytics System<br><br>
                    <span style="font-size: 10px; color: grey;">Note: This is an automated message. Please do not reply.</span>
                    </p>
                </body>
            </html>        
            """

        body.add_alternative(html_msg, subtype='html')

        json_file_path = content.get("filepath")
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'rb') as f:
                json_content = f.read()
            attachment = MIMEApplication(json_content)
            attachment.add_header('Content-Disposition', f'attachment; filename="{scan_id}.json"')
            body.attach(attachment)
            
        calibration_log_path = f"/home/pi/OakRaman2/assets/scanner_config/{device_id}_calibration_assets.txt"
        if os.path.exists(calibration_log_path):
            with open(calibration_log_path, 'rb') as txt_file:
                txt_content = txt_file.read()
            txt_attachment = MIMEApplication(txt_content)
            txt_attachment.add_header('Content-Disposition', f'attachment; filename="{device_id}_calibration_assets.txt"')
            body.attach(txt_attachment)

        return body
    
    
    def _create_product_scan_data_sync_email_body(self, content):
        sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
        scan_list = content.get("scan_list") or []  # Ensures scan_list is a list
        print("scan_list", scan_list)
        device_id = content.get('device', 'Unknown')
        body = EmailMessage()

        if len(scan_list) == 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} product Scan Sync Request [Scan ID: {scan_list[0]}]"
            scan_id = scan_list[0]
        elif len(scan_list) > 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} All product Scan Sync Request [Scan IDs: {scan_list[0]} to {scan_list[-1]}]"
            scan_id = scan_list[0]
        else:
            body['Subject'] = f"Oak Analytics - NR-{device_id} product Scan Sync Request [No Scan ID]"
            scan_id = "scan_data"

        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        # body['Cc'] = self.__cred['cc']

        scan_id_list_html = ''.join(f"<li>{s}</li>" for s in scan_list) if scan_list else "<li>No scans available</li>"

        html_msg = f""" 
            <html>
                <body>
                    <p>Hello Team,<br><br>
                    This is an automated product scan sync request from device <strong>NR-{device_id}</strong>.<br><br>
                    <strong>Scan Details:</strong>
                    <ul>
                        <li>Scan Count: {len(scan_list)}</li>
                    </ul>
                    <strong>Scan IDs:</strong>
                    <ul>
                        {scan_id_list_html}
                    </ul>
                    Attached JSON contains the product scan data for syncing to AWS.<br><br>
                    Regards,<br>
                    Oak Analytics System<br><br>
                    <span style="font-size: 10px; color: grey;">Note: This is an automated message. Please do not reply.</span>
                    </p>
                </body>
            </html>        
            """

        body.add_alternative(html_msg, subtype='html')

        json_file_path = content.get("filepath")
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'rb') as f:
                json_content = f.read()
            attachment = MIMEApplication(json_content)
            attachment.add_header('Content-Disposition', f'attachment; filename="{scan_id}.json"')
            body.attach(attachment)

        return body
    
    def _create_scan_notes_data_sync_email_body(self, content):
        sender_email_style = 'font-weight: bold; color: blue; font-size: 16px;'
        scan_list = content.get("scan_list") or []  # Ensures scan_list is a list
        print("scan_list", scan_list)
        device_id = content.get('device', 'Unknown')
        body = EmailMessage()

        if len(scan_list) == 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} Scan Notes Sync Request [Scan ID: {scan_list[0]}]"
            scan_id = scan_list[0]
        elif len(scan_list) > 1:
            body['Subject'] = f"Oak Analytics - NR-{device_id} All Scan Notes Sync Request [Scan IDs: {scan_list[0]} to {scan_list[-1]}]"
            scan_id = scan_list[0]
        else:
            body['Subject'] = f"Oak Analytics - NR-{device_id} Scan Notes Sync Request [No Scan ID]"
            scan_id = "scan_data"

        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        # body['Cc'] = self.__cred['cc']

        scan_id_list_html = ''.join(f"<li>{s}</li>" for s in scan_list) if scan_list else "<li>No scans available</li>"

        html_msg = f""" 
            <html>
                <body>
                    <p>Hello Team,<br><br>
                    This is an automated scan notes sync request from device <strong>NR-{device_id}</strong>.<br><br>
                    <strong>Scan Details:</strong>
                    <ul>
                        <li>Scan Count: {len(scan_list)}</li>
                    </ul>
                    <strong>Scan IDs:</strong>
                    <ul>
                        {scan_id_list_html}
                    </ul>
                    Attached JSON contains the product scan data for syncing to AWS.<br><br>
                    Regards,<br>
                    Oak Analytics System<br><br>
                    <span style="font-size: 10px; color: grey;">Note: This is an automated message. Please do not reply.</span>
                    </p>
                </body>
            </html>        
            """

        body.add_alternative(html_msg, subtype='html')

        json_file_path = content.get("filepath")
        if json_file_path and os.path.exists(json_file_path):
            with open(json_file_path, 'rb') as f:
                json_content = f.read()
            attachment = MIMEApplication(json_content)
            attachment.add_header('Content-Disposition', f'attachment; filename="{scan_id}.json"')
            body.attach(attachment)

        return body

    def __create_callibEmailBody(self, content):
        body = EmailMessage()
        body['Subject'] = f'Calibration Required for Device NR_{content.get("device")}'
        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        body['Bcc'] = self.__cred['bcc']

        html_msg = f""" 
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href='https://fonts.googleapis.com/css?family=Nunito:400,300' rel='stylesheet' type='text/css'>
        </head>
        <body>
            <p>Dear Team,<br/><br/>
            Calibration is required for Device NR_{content.get("device")}.<br/><br/>
            Thanks & Regards,<br/>
            {content.get('customer')}<br/><br/>
            <span style='font-weight: bold; color: grey; font-size: 10px;'>
            Please note: This is an automated email—no reply is needed.</span>
            </p>
        </body>
        </html>"""

        body.add_alternative(html_msg, subtype='html')
        return body
    
    def __create_streydetectEmailBody(self, content):
        body = EmailMessage()
        body['Subject'] = f'Stray light error detected in Device NR_{content.get("device")}'
        body['From'] = self.__cred["sender_mail"]
        body['To'] = self.__cred['receiver_mail']
        body['Bcc'] = self.__cred['bcc']

        html_msg = f""" 
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href='https://fonts.googleapis.com/css?family=Nunito:400,300' rel='stylesheet' type='text/css'>
        </head>
        <body>
            <p>Dear Team,<br/><br/>
            Stray light error detected in Device NR_{content.get("device")}.<br/><br/>
            Thanks & Regards,<br/>
            {content.get('customer')}<br/><br/>
            <span style='font-weight: bold; color: grey; font-size: 10px;'>
            Please note: This is an automated email—no reply is needed.</span>
            </p>
        </body>
        </html>"""

        body.add_alternative(html_msg, subtype='html')
        image_folder = content.get("image_path")  
    # example:
    # /home/pi/OakRaman2/assets/scanner_data/scan/scanimages/PrechecksprayIDMaterial/...

        if image_folder and os.path.isdir(image_folder):
            for filename in os.listdir(image_folder):
                file_path = os.path.join(image_folder, filename)

                if os.path.isfile(file_path):
                    ctype, encoding = mimetypes.guess_type(file_path)
                    if ctype is None or encoding is not None:
                        ctype = 'application/octet-stream'

                    maintype, subtype = ctype.split('/', 1)

                    with open(file_path, 'rb') as f:
                        body.add_attachment(
                            f.read(),
                            maintype=maintype,
                            subtype=subtype,
                            filename=filename
                        )
        return body
    

    def send_mail(self,content={},mail_type='support'):
            response = 0
            if self.__smtp_server==0 or not self.check_session(self.__smtp_server):
                self.__create_session() 
                
            if mail_type=='support': 
                body = self.__create_supportEmailBody(content=content)
            else: body = self.__create_EmailBody(content=content)
            
            try:
                    response = self.__smtp_server.sendmail(self.__cred['sender_mail'],[self.__cred['receiver_mail']],body.as_string())   
            except Exception as e:
                    print(f"Failed to send mail - {e}")
            if response=={}: 
                return True
            else: 
                return False
            
    def quit_server(self):
        if self.__smtp_server!=0:
            if self.__smtp_server.quit()[0]==221:
                print(f"Email_Server_Terminated at time = {datetime.datetime.now()} ")
        else:
            print("Email Server not created")

    def send_mail2(self, content={}, mail_type='support'):
            response = 0
            if self.__smtp_server == 0 or not self.check_session(self.__smtp_server):
                self.__create_session() 
                
            if mail_type == 'support': 
                body = self.__create_supportEmailBody(content=content)

            elif mail_type == "followup_support":
               body = self.__followup_support(content=content)

            elif mail_type == "calibration":
                body =  self.__create_callibEmailBody(content=content)
            elif mail_type == "stray_light":
                body = self.__create_streydetectEmailBody(content=content)
            elif mail_type =='scan_sync':
                body =  self._create_id_sync_email_body(content=content)
            else:
                body = self.__create_EmailBody(content=content)

            try:
                # Extract CC emails and split them into a list
                cc_recipients = self.__cred.get('bcc', '')  
                cc_list = [email.strip() for email in cc_recipients.split(",") if email.strip()]

                # Combine TO and CC recipients
                all_recipients = [self.__cred['receiver_mail']] + cc_list

                # Send the email with TO + CC recipients
                response = self.__smtp_server.sendmail(self.__cred['sender_mail'], all_recipients, body.as_string())   
            except Exception as e:
                print(f"Failed to send mail - {e}")
            
            if response=={}:
                return True
            else: 
                return False
            
    def send_mail3(self, content={}, mail_type='scan_sync'):
        response = 0
        if self.__smtp_server == 0 or not self.check_session(self.__smtp_server):
            self.__create_session()

        if mail_type == 'scan_sync':
            body = self._create_id_sync_email_body(content=content)

        elif mail_type == 'xcal':
           body = self._create_callibration_data_sync_email_body(content=content)
        
        elif mail_type == 'product':
            body = self._create_product_scan_data_sync_email_body(content)
        elif mail_type == 'notes':
            body = self._create_scan_notes_data_sync_email_body(content)
        
        else:
            body = self.__create_EmailBody(content=content)

        try:
            # Send the email to the primary recipient only
            response = self.__smtp_server.sendmail(
                self.__cred['sender_mail'],
                [self.__cred['receiver_mail']],
                body.as_string()
            )
        except Exception as e:
            print(f"Failed to send mail - {e}")

        if response == {}:
            return True
        else:
            return False


def expert_review(content,on='identification'):
    try:
        if on=='identification':
            er_bcc = get_customer()
            cred = {
            'host': host,
            "port": port,
            'sender_mail': expert_sender,
            'receiver_mail': receiver_mail_chat,
            'bcc': er_bcc.get('ER_bcc'),
            'NameServer': expert_NameServer,
            'passcode': expert_passcode,
            'quit_time': 10}
            email_SS =  Email_Sender(cred=cred)
            email_SS.send_mail2(content=content,mail_type='support')
            email_SS.quit_server()
            return True
        ###### TO DO ######
        """
            if on == 'quantiifcation': write up support mail body 
        """
    except Exception as e:
        print("Email Sent Failed: ",e.__doc__)
        return False


def callibrartion_mail_sent(content):
    try:
        cal = get_customer()
        cred = {
        'host': host,
        "port": port,
        'sender_mail': expert_sender,
        'receiver_mail': cal.get("to"),
        'bcc':cal.get('calibration_bcc'),
        'NameServer': expert_NameServer,
        'passcode': expert_passcode,
        'quit_time': 10}
        email_SS =  Email_Sender(cred=cred)
        email_SS.send_mail2(content=content,mail_type='calibration')
        email_SS.quit_server()
        return True
    except Exception as e:
        print("Email Sent Failed: ",e.__doc__)
        return False
    
def data_sync_mail_sent(content,mail_type='scan_sync'):
    try:
        cred = {
        'host': host,
        "port": port,
        'sender_mail': expert_sender,
        'receiver_mail': receiver_mail_chat,
        # 'cc':" ",
        'NameServer': expert_NameServer,
        'passcode': expert_passcode,
        'quit_time': 10}
        email_SS =  Email_Sender(cred=cred)
        email_SS.send_mail3(content=content,mail_type=mail_type)
        email_SS.quit_server()
        return True
    except Exception as e:
        print("Email Sent Failed: ",e.__doc__)
        return False
    
def stray_light_mail_sent(content):
    try:
        cal = get_customer()
        cred = {
        'host': host,
        "port": port,
        'sender_mail': expert_sender,
        'receiver_mail': cal.get("to"),
        'bcc':cal.get('calibration_bcc'),
        'NameServer': expert_NameServer,
        'passcode': expert_passcode,
        'quit_time': 10}
        email_SS =  Email_Sender(cred=cred)
        email_SS.send_mail2(content=content,mail_type='stray_light')
        email_SS.quit_server()
        return True
    except Exception as e:
        print("Email Sent Failed: ",e.__doc__)
        return False 
############ 
def send_customer_followup(scan_id, device_id, user_email):
    """
    Sends a follow-up email to the customer informing them that the team is getting in touch.
    """
    try:
        cred = {
            'host': host,
            "port": port,
            'sender_mail': support_sender_mail,  # Sending from Support Email
            'receiver_mail': user_email,
            #'cc':"deepak@oakanalytics.com, shivali.krishna@nutanxt.com, pranita.zaware@nutanxt.com, ajay.k@nutanxt.com, shubham.g@nutanxt.com, shubham.purkar@nutanxt.com, vaishnav.shevale@nutanxt.com",
            'passcode': support_passcode,
            'NameServer': support_NameServer,
            'quit_time': 10
        }

        email_SS = Email_Sender(cred=cred)

        # Email Subject
        subject = f"Follow-up: We are reviewing your request (Scan ID: {scan_id})"

        # Email Body
        html_body = f"""
        <html>
        <body>
            <p>Hello,</p>
            <p>We have received your request for an Expert Review.</p>
            <p><strong>Scan ID:</strong> {scan_id}</p>
            <p><strong>Device ID:</strong> {device_id}</p>
            <p>Our team is analyzing your request, and we will get back to you with the Expert Review.</p>
            <p>Thank you for reaching out.</p>
            <br/>
            <p>Best Regards,<br/>Support Team - NarcRanger</p>
        </body>
        </html>
        """



        content = {
            "subject": subject,
            "user_id": user_email,
            "scan_id": scan_id,
            "device": device_id,
            "html": html_body
        }

        # Send Email
        email_SS.send_mail2(content=content, mail_type='followup_support')
        email_SS.quit_server()
        print(f"Follow-up email sent to {user_email} for Scan ID {scan_id}")

    except Exception as e:
        print(f"Failed to send follow-up email: {e}")
        