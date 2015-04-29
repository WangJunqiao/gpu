package dd.crawler.codeforce.statusparser;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.htmlparser.Parser;
import org.htmlparser.Remark;
import org.htmlparser.Tag;
import org.htmlparser.Text;
import org.htmlparser.visitors.NodeVisitor;

import dd.crawler.codeforce.meta.Status;

public class StatusParser {

	public static List<Status> list;


	public static NodeVisitor myVisitor = new NodeVisitor() {

		/** 重载抽象类NodeVisitor的beginParsing方法,解析开始时调用此方法 */
		public void beginParsing() {
			//System.out.println("开始解析HTML内容......");
		}

		/** 重载抽象类NodeVisitor的finishedParsing方法,解析结束时调用此方法 */
		public void finishedParsing() {
			//System.out.println("整个HTML内容解析完毕!");
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

		/** 重载抽象类NodeVisitor的visitTag方法,遇到开始标签时调用此方法 */
		public void visitTag(Tag tag) {
			//System.out.println("开始当前标签: " + tag.getText());
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

		/** 重载抽象类NodeVisitor的visitEndTag方法,遇到结束标签时调用此方法 */
		public void visitEndTag(Tag tag) {
			//System.out.println("结束当前标签: " + tag.getText());
		}

		/** 重载抽象类NodeVisitor的visitStringNode方法,遇到文本节点时调用此方法 */
		public void visitStringNode(Text string) {
			//System.out.println("当前文本节点: " + string);
		}

		/** 重载抽象类NodeVisitor的visitRemarkNode方法,遇到注释时调用此方法 */
		public void visitRemarkNode(Remark remark) {
			//System.out.println("当前注释: " + remark);
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
