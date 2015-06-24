package edu.zju.dd.codeforce.statusparser;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.htmlparser.Parser;
import org.htmlparser.Remark;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.visitors.NodeVisitor;

import edu.zju.dd.codeforce.meta.Status;

public class StatusParser {

	public static List<Status> list;


	public static NodeVisitor myVisitor = new NodeVisitor() {

		/** ���س�����NodeVisitor��beginParsing����,������ʼʱ���ô˷��� */
		public void beginParsing() {
			//System.out.println("��ʼ����HTML����......");
		}

		/** ���س�����NodeVisitor��finishedParsing����,��������ʱ���ô˷��� */
		public void finishedParsing() {
			//System.out.println("���HTML���ݽ������!");
		}

		private int getFirstNumber(String s) {
			int ret = 0;
			for (int i = 0; i < s.length(); i++) {
				char ch = s.charAt(i);
				if (ch >= '0' && ch <= '9') {
					ret = ret * 10 + (ch - '0');
				} else {
					break;
				}
			}
			return ret;
		}

		/** ���س�����NodeVisitor��visitTag����,������ʼ��ǩʱ���ô˷��� */
		public void visitTag(Tag tag) {
			//System.out.println("��ʼ��ǰ��ǩ: " + tag.getText());
			//if(tag.getText().equals(arg0))
			
			if (tag.getText().indexOf("data-submission-id=") != -1) {
				Scanner scanner = new 
						Scanner(tag.toPlainTextString());
				List<String> para = new ArrayList<String>();
				while (scanner.hasNext()) {
					String line = scanner.nextLine().trim();
					if (line.equals(""))
						continue;
					para.add(line);
				}
				if(para.get(0).charAt(0)<'0' || para.get(0).charAt(0)>'9')
					return;
				Status status = new Status();
				status.setSubmissionId(para.get(0));
				status.setTimestamp(para.get(1));
				status.setCoderId(para.get(2));
				status.setContestId(getFirstNumber(para.get(3)) + "");
				status.setLanguage(para.get(4));
				status.setResult(para.get(5));
				status.setRunTime(para.get(6));
				status.setRunMemory(para.get(7));
				list.add(status);
			}
		}

		/** ���س�����NodeVisitor��visitEndTag����,���������ǩʱ���ô˷��� */
		public void visitEndTag(Tag tag) {
			//System.out.println("����ǰ��ǩ: " + tag.getText());
		}

		/** ���س�����NodeVisitor��visitStringNode����,�����ı��ڵ�ʱ���ô˷��� */
		public void visitStringNode(Text string) {
			//System.out.println("��ǰ�ı��ڵ�: " + string);
		}

		/** ���س�����NodeVisitor��visitRemarkNode����,����ע��ʱ���ô˷��� */
		public void visitRemarkNode(Remark remark) {
			//System.out.println("��ǰע��: " + remark);
		}
	};

	public static List<Status> parseStatusFromUrl(String urlString) throws Exception {
		list = new ArrayList<Status>();
		URL url = new URL(urlString);
		Parser parser = new Parser(url.openConnection());
		parser.visitAllNodesWith(myVisitor);
		return list;
	}

	public static List<Status> parseStatusFromHtml(String html) throws Exception {
		list = new ArrayList<Status>();
		Parser parser = new Parser(html);
		parser.visitAllNodesWith(myVisitor);
		return list;
	}
}
