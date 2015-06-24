package edu.zju.dd.codeforce.statusparser;

import java.net.URL;

import org.htmlparser.Parser;
import org.htmlparser.Remark;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.visitors.NodeVisitor;



public class SubmissionParser {
	private boolean code_found;
	private String code;

	private NodeVisitor myVisitor = new NodeVisitor() {

		/** ���س�����NodeVisitor��beginParsing����,������ʼʱ���ô˷��� */
		public void beginParsing() {
			//System.out.println("��ʼ����HTML����......");
		}

		/** ���س�����NodeVisitor��finishedParsing����,��������ʱ���ô˷��� */
		public void finishedParsing() {
			//System.out.println("���HTML���ݽ������!");
		}

		/** ���س�����NodeVisitor��visitTag����,������ʼ��ǩʱ���ô˷��� */
		public void visitTag(Tag tag) {
			//System.out.println("��ʼ��ǰ��ǩ: " + tag.getText());
			//if(tag.getText().equals(arg0))
			
			if (tag.getText().indexOf("class=\"prettyprint\"") != -1) {
				code_found = true;
			}
		}

		/** ���س�����NodeVisitor��visitEndTag����,���������ǩʱ���ô˷��� */
		public void visitEndTag(Tag tag) {
			//System.out.println("����ǰ��ǩ: " + tag.getText());
		}

		/** ���س�����NodeVisitor��visitStringNode����,�����ı��ڵ�ʱ���ô˷��� */
		public void visitStringNode(Text string) {
			//System.out.println("��ǰ�ı��ڵ�: " + string.toPlainTextString());
			if(code_found) {
				code = string.toPlainTextString();
				code_found = false;
			}
			//StringBuilder sb = new StringBuilder(string);
			
		}

		/** ���س�����NodeVisitor��visitRemarkNode����,����ע��ʱ���ô˷��� */
		public void visitRemarkNode(Remark remark) {
			//System.out.println("��ǰע��: " + remark);
		}
	};
	
	
	public String parseCodeFromUrl(String urlString) throws Exception {
		code_found = false;
		URL url = new URL(urlString);
		Parser parser = new Parser(url.openConnection());
		parser.visitAllNodesWith(myVisitor);
		return code;
	}
	
	public String parseCodeFromHtml(String html) throws Exception {
		Parser parser = new Parser(html);
		code_found = false;
		parser.visitAllNodesWith(myVisitor);
		return code;
	}
}
